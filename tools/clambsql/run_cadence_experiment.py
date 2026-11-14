"""运行 CADENCE 在 CLAMBSQL 候选 SQL 池上的交互评测。

该入口读取官方 ``ambiguous_queries``，由方法核心控制候选分组和 DP probe 选择；
回答默认使用确定性 Gold-oracle，也可切换为 Gold-conditioned LLM 用户模拟。
请求可并发执行，结果仍按输入顺序追加到 JSONL，支持从连续检查点恢复。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_CADENCE_ROOT = _SOURCE_ROOT

from cadence.datasets.clambsql import ClambsqlRecord, load_clambsql_dataset  # noqa: E402
from cadence.datasets.common import sha256_of_file  # noqa: E402
from cadence.factorize_sql import (  # noqa: E402
    CadenceSQLFactorizationError,
    factorize_sql,
)
from cadence.grouping import group_equivalent_candidates  # noqa: E402
from cadence.interaction import (  # noqa: E402
    CadenceFeedbackInput,
    CadenceMemoryBinding,
    run_cadence_interaction,
)
from cadence.memory import (  # noqa: E402
    E5_ENCODER_ID,
    MemoryConfig,
    MemoryEngine,
    MemoryScope,
    MultilingualE5SmallEncoder,
    PreferenceEventV1,
    SQLiteMemoryStore,
)
from cadence.parse_bind import SchemaCatalog  # noqa: E402
from cadence.provider import call_role  # noqa: E402
from cadence.runtime_config import (  # noqa: E402
    RuntimeConfig,
    load_runtime_config,
    require_api_key_env,
    resolve_memory_runtime_config,
    resolve_runtime_role,
)
from tools.cadence.llm_gold_oracle import LLMGoldOptionOracle  # noqa: E402
from tools.clambsql.cadence_callbacks import CadenceCallbacks  # noqa: E402
from tools.clambsql.cadence_metrics import build_cadence_metrics  # noqa: E402

_RUN_SCHEMA = "CadenceCLAMBSQLRun"
_CADENCE_RUNTIME_ROLE = {
    "cadence_attribution": "utterance_attribution",
    "cadence_option_assessment": "utterance_attribution",
    "cadence_question": "clarification_wording",
    "cadence_gold_simulation": "gold_user_simulation",
}
_METHOD_CORE_PATHS = (
    "src/cadence/constants.py",
    "src/cadence/candidate_pool.py",
    "src/cadence/grouping.py",
    "src/cadence/expected_cost.py",
    "src/cadence/pipeline.py",
    "src/cadence/factorize_sql.py",
    "src/cadence/relational_algebra.py",
    "src/cadence/interaction.py",
    "src/cadence/memory_policy.py",
    "src/cadence/memory/model.py",
    "src/cadence/memory/engine.py",
    "src/cadence/memory/event_log.py",
    "src/cadence/memory/sqlite_store.py",
    "src/cadence/memory/encoders.py",
    "src/cadence/parse_bind.py",
    "src/cadence/sqlglot_ast_mapper.py",
    "src/cadence/canonical_serialize.py",
)
_EXPERIMENT_SOURCE_PATHS = (
    "tools/clambsql/cadence_callbacks.py",
    "tools/clambsql/run_cadence_experiment.py",
    "tools/clambsql/cadence_metrics.py",
    "tools/clambsql/sqlite_execution.py",
    "src/cadence/provider.py",
    "src/cadence/runtime_config.py",
    "src/cadence/datasets/common.py",
    "src/cadence/datasets/clambsql.py",
    "tools/cadence/llm_gold_oracle.py",
)


def _record_schema_id(record: ClambsqlRecord) -> str:
    """用 catalog 与显式时间列元数据生成稳定的 schema 隔离键。"""

    schema_text = json.dumps(
        {
            "catalog": record.request.catalog,
            "temporal_columns": [list(pair) for pair in record.request.temporal_columns],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(schema_text.encode("utf-8")).hexdigest()


class CompletionWithStats(Protocol):
    """完成一轮模型请求，并只暴露不含提示词的调用摘要。"""

    calls: list[dict[str, Any]]

    def __call__(
        self, callback_role: str, messages: Sequence[Mapping[str, str]]
    ) -> str: ...


class RuntimeCompletion:
    """通过现有 runtime provider 发起 Flash 调用并累计不含 prompt 的使用统计。"""

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        self._runtime_config = runtime_config
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, callback_role: str, messages: Sequence[Mapping[str, str]]
    ) -> str:
        """按回调职责映射到已配置的模型运行角色。"""

        runtime_role = _CADENCE_RUNTIME_ROLE.get(callback_role)
        if runtime_role is None:
            raise ValueError("CADENCE_CALLBACK_ROLE_UNKNOWN")
        content, metadata = call_role(self._runtime_config, runtime_role, messages)
        if not isinstance(content, str):
            raise ValueError("CADENCE_CALLBACK_RESPONSE_INVALID")
        self.calls.append(
            {
                "callback_role": callback_role,
                "runtime_role": runtime_role,
                "model_id": metadata.get("model_id"),
                "model": metadata.get("model"),
                "token_usage": metadata.get("token_usage"),
            }
        )
        return content


@dataclass(frozen=True)
class MemoryRuntime:
    """本次 CLAMBSQL 运行的记忆存储、阈值与稳定会话时间。"""

    store: SQLiteMemoryStore
    engine: MemoryEngine
    config: MemoryConfig
    m_min: float
    session_id: str
    session_start: datetime

    def binding_for(self, record: ClambsqlRecord) -> CadenceMemoryBinding:
        """按数据库和 schema 隔离偏好，并用稳定时间支持断点恢复。"""

        scope = MemoryScope(
            namespace="EXPERIMENT:CLAMBSQL",
            user_id="clambsql-simulated-user",
            database_id=f"{record.db_id}/{record.db_file}",
            schema_id=_record_schema_id(record),
        )
        return CadenceMemoryBinding(
            engine=self.engine,
            scope=scope,
            config=self.config,
            m_min=self.m_min,
            as_of=self.session_start + timedelta(seconds=record.index),
        )


def _stable_json_hash(value: Mapping[str, Any]) -> str:
    """对 JSON 字段排序后散列，生成事件账本使用的稳定 ID。"""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _memory_store_path(
    runtime_config: RuntimeConfig,
    *,
    output_dir: Path,
    repo_root: Path,
) -> tuple[Mapping[str, Any] | None, Path | None]:
    """将 Memory 开关和存储路径解析为显式运行输入。"""

    settings = resolve_memory_runtime_config(runtime_config)
    if settings is None:
        return None, None
    enabled = settings.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("CADENCE_MEMORY_CONFIG_INVALID")
    if not enabled:
        return settings, None
    storage = settings.get("storage")
    if not isinstance(storage, Mapping):
        raise ValueError("CADENCE_MEMORY_CONFIG_INVALID")
    if storage.get("format") != "PreferenceSQLiteVecStoreV1":
        raise ValueError("CADENCE_MEMORY_STORE_FORMAT_UNSUPPORTED")
    configured_path = storage.get("path")
    if configured_path is None:
        return settings, (output_dir / "memory.sqlite").resolve()
    if not isinstance(configured_path, str) or not configured_path.strip():
        raise ValueError("CADENCE_MEMORY_STORE_PATH_INVALID")
    path = Path(configured_path)
    if not path.is_absolute():
        path = repo_root / path
    return settings, path.resolve()


def _prepare_memory_runtime(
    settings: Mapping[str, Any] | None,
    *,
    store_path: Path | None,
    session_start: datetime | None,
) -> MemoryRuntime | None:
    """只在配置启用时构造记忆编码器、参数、SQLite store 和引擎。"""

    if settings is None or settings.get("enabled") is False:
        return None
    if store_path is None or session_start is None:
        raise ValueError("CADENCE_MEMORY_CONFIG_INVALID")
    retrieval = settings.get("retrieval")
    scoring = settings.get("scoring")
    decision = settings.get("decision")
    if not all(isinstance(item, Mapping) for item in (retrieval, scoring, decision)):
        raise ValueError("CADENCE_MEMORY_CONFIG_INVALID")
    assert isinstance(retrieval, Mapping)
    assert isinstance(scoring, Mapping)
    assert isinstance(decision, Mapping)
    if (
        retrieval.get("encoder_id") != E5_ENCODER_ID
        or scoring.get("method") != "EXPONENTIAL_TIME_DECAY"
        or scoring.get("age_unit") != "UTC_CALENDAR_DAYS"
        or decision.get("policy") != "GLOBAL_TWO_THRESHOLD"
    ):
        raise ValueError("CADENCE_MEMORY_CONFIG_UNSUPPORTED")
    aliases = retrieval.get("aliases", {})
    if not isinstance(aliases, Mapping):
        raise ValueError("CADENCE_MEMORY_CONFIG_INVALID")
    try:
        config = MemoryConfig(
            encoder_id=E5_ENCODER_ID,
            tau_retrieval=float(retrieval["tau_r"]),
            half_life_days=float(scoring["half_life_days"]),
            min_effective_history=None,
            tau_auto=float(decision["tau_auto"]),
            semantic_top_k=int(retrieval["semantic_top_k"]),
            parameter_set_id=str(decision["parameter_set_id"]),
            calibration_id=decision.get("calibration_id"),
            validation_fingerprint=decision.get("validation_fingerprint"),
        )
        m_min_raw = decision["m_min"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("CADENCE_MEMORY_CONFIG_INVALID") from exc
    if (
        isinstance(m_min_raw, bool)
        or not isinstance(m_min_raw, (int, float))
        or m_min_raw < 0
    ):
        raise ValueError("CADENCE_MEMORY_CONFIG_INVALID")
    encoder = MultilingualE5SmallEncoder()
    if encoder.encoder_id != E5_ENCODER_ID:
        raise ValueError("CADENCE_MEMORY_ENCODER_MISMATCH")
    store = SQLiteMemoryStore(store_path)
    engine = MemoryEngine(store, encoder=encoder, aliases=aliases)
    session_value = session_start.astimezone(UTC).isoformat().replace("+00:00", "Z")
    session_id = _stable_json_hash({"session_start": session_value})
    return MemoryRuntime(
        store=store,
        engine=engine,
        config=config,
        m_min=float(m_min_raw),
        session_id=session_id,
        session_start=session_start.astimezone(UTC),
    )


def _memory_event(
    feedback: CadenceFeedbackInput,
    *,
    record: ClambsqlRecord,
    runtime: MemoryRuntime,
) -> dict[str, object]:
    """把明确的 Gold-oracle 选项回答保存为可重放的偏好事件。"""

    mapping = feedback.attribution.option_mapping
    chosen = next(
        (item for item in mapping.options if item.option_id == feedback.chosen_option_id),
        None,
    )
    if chosen is None or not feedback.candidate_ids:
        raise ValueError("CADENCE_MEMORY_FEEDBACK_INVALID")
    attribution_identity = _stable_json_hash(
        {
            "ambiguity_text": feedback.attribution.ambiguity_text,
            "context_key": feedback.attribution.context_key,
            "context_text": feedback.attribution.context_text,
            "factor_type": feedback.partition.factor_type,
            "scope_id": feedback.partition.scope_id,
            "slot_id": feedback.partition.slot_id,
            "options": [
                {
                    "option_id": item.option_id,
                    "preference_value_id": item.preference_value_id,
                    "interpretation": feedback.attribution.option_interpretations[
                        item.option_id
                    ],
                    "intent_assessment": feedback.attribution.option_intent_assessments[
                        item.option_id
                    ],
                }
                for item in mapping.options
            ],
        }
    )
    occurred_at = (feedback.as_of or runtime.session_start) + timedelta(
        microseconds=feedback.round_index
    )
    occurred_text = occurred_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    request_id = f"clambsql-{record.index}"
    event_id = _stable_json_hash(
        {
            "session_id": runtime.session_id,
            "request_id": request_id,
            "round_index": feedback.round_index,
            "probe_id": feedback.partition.partition_id,
            "attribution_id": attribution_identity,
            "chosen_option_id": chosen.option_id,
        }
    )
    option_assessment = feedback.option_assessment
    evidence = json.dumps(
        {
            "semantic_relevance": option_assessment.semantic_relevance,
            "semantic_distinguishability": (
                option_assessment.semantic_distinguishability
            ),
            "user_answerability": option_assessment.user_answerability,
            "intent_assessments": dict(
                sorted(feedback.attribution.option_intent_assessments.items())
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload: dict[str, Any] = {
        "schema_version": "PreferenceEventV1",
        "event_id": event_id,
        "namespace": "EXPERIMENT:CLAMBSQL",
        "scope": {
            "user_id": "clambsql-simulated-user",
            "database_id": f"{record.db_id}/{record.db_file}",
            "schema_id": _record_schema_id(record),
        },
        "retrieval_text": {
            "ambiguity_key": feedback.partition.factor_type,
            "ambiguity_text": feedback.attribution.ambiguity_text,
            "context_key": feedback.attribution.context_key,
            "context_text": feedback.attribution.context_text,
        },
        "preference": {
            "preference_schema_version": "PreferenceValueV1",
            "preference_value_id": chosen.preference_value_id,
            "preference_value_text": feedback.attribution.option_interpretations[
                chosen.option_id
            ],
        },
        "feedback_action": "ASK_OPTION_SELECTED",
        "admission": {
            "memory_admission_assessment": {
                "user_distinguishability": "DISTINGUISHABLE",
                "validation_status": "ASSESSED",
                "evidence": evidence,
            },
            "answer_source": "ORACLE_SIMULATED",
            "answer_kind": "OPTION",
            "grounding_status": "GROUNDED_OPTION",
            "reliably_committed": True,
        },
        "mapping_snapshot": {
            "candidate_set_key": mapping.candidate_set_key,
            "candidate_ids": list(feedback.candidate_ids),
            "option_mappings": [
                {
                    "option_id": item.option_id,
                    "preference_value_id": item.preference_value_id,
                    "mapped_node_ids": list(item.mapped_node_ids),
                }
                for item in mapping.options
            ],
            "chosen_option_id": chosen.option_id,
            "recommended_option_id": None,
            "recommended_preference_value_id": None,
            "mapped_node_ids": list(chosen.mapped_node_ids),
        },
        "source_refs": {
            "request_id": request_id,
            "session_id": runtime.session_id,
            "round_index": feedback.round_index,
            "probe_id": feedback.partition.partition_id,
            "memory_decision_id": (
                feedback.memory_decision.decision_id
                if feedback.memory_decision is not None
                else None
            ),
            "answer_evidence_id": None,
            "attribution_id": attribution_identity,
        },
        "occurred_at": occurred_text,
    }
    return PreferenceEventV1.from_mapping(payload).to_mapping()


def _source_fingerprint(repo_root: Path, paths: Sequence[str]) -> str:
    """绑定指定核心或实验外壳文件的原始字节，不混淆二者来源。"""

    digest = hashlib.sha256()
    for relative in paths:
        path = repo_root / Path(relative)
        data = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """临时文件替换进度快照，避免中断留下半个 JSON。"""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _manifest_path(path: Path, repo_root: Path) -> str:
    """源数据可在副本之外保存；清单对仓库内路径用相对值、外部路径用绝对值。"""

    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _read_completed(path: Path) -> dict[int, Mapping[str, Any]]:
    """恢复时拒绝重复、越界或非连续的请求记录，防止漏行伪装成完整结果。"""

    completed: dict[int, Mapping[str, Any]] = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("CADENCE_RUN_CHECKPOINT_INVALID") from exc
        if not isinstance(row, Mapping):
            raise ValueError("CADENCE_RUN_CHECKPOINT_INVALID")
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index in completed:
            raise ValueError("CADENCE_RUN_CHECKPOINT_INVALID")
        if "final_sql" not in row or "predicted_ambiguities" not in row:
            raise ValueError("CADENCE_RUN_CHECKPOINT_INVALID")
        completed[index] = row
    return completed


def _factorize_record(record: ClambsqlRecord) -> tuple[Any, ...]:
    """将官方候选转换为 CADENCE 因子表示，不补写或过滤候选 SQL。"""

    catalog = SchemaCatalog.from_definition(record.request.catalog)
    factorized = []
    for candidate in record.request.candidates:
        candidate_id = candidate["candidate_id"]
        sql_text = candidate["sql_text"]
        if not isinstance(candidate_id, str) or not isinstance(sql_text, str):
            raise ValueError("CADENCE_RUN_CANDIDATE_INVALID")
        factorized.append(
            factorize_sql(
                sql=sql_text,
                candidate_id=candidate_id,
                catalog=catalog,
                dialect="sqlite",
                temporal_columns=record.request.temporal_columns,
            )
        )
    return tuple(factorized)


def run_one_record(
    record: ClambsqlRecord,
    *,
    completion: CompletionWithStats,
    memory_runtime: MemoryRuntime | None = None,
    answer_mode: str = "gold_oracle",
) -> dict[str, Any]:
    """执行单条请求，并保存逐轮可见选项、类别映射与用户模拟选择。"""

    candidate_sql = {
        str(candidate["candidate_id"]): str(candidate["sql_text"])
        for candidate in record.request.candidates
    }
    gold_candidate_ids = {
        candidate_id
        for candidate_id, sql_text in candidate_sql.items()
        if sql_text == record.gold_query
    }
    trace_gold_class_ids: set[str] = set()
    trace_gold_class_id: str | None = None
    interaction_trace: list[dict[str, Any]] = []
    base: dict[str, Any] = {
        "index": record.index,
        "final_sql": None,
        "predicted_ambiguities": [],
        "interaction_status": "ABSTAIN",
        "reason_code": None,
        "questions_asked": 0,
        "selected_candidate_ids": [],
        "candidate_pool_contains_gold": bool(gold_candidate_ids),
        "answer_mode": answer_mode,
        "interaction_trace": interaction_trace,
    }

    def append_final_trace(
        *,
        status: str,
        reason_code: str | None,
        selected_candidate_ids: Sequence[str] = (),
    ) -> None:
        selected_ids = set(selected_candidate_ids)
        interaction_trace.append(
            {
                "schema_version": "CadenceInteractionTraceEventV1",
                "event_type": "INTERACTION_END",
                "request_index": record.index,
                "answer_mode": answer_mode,
                "status": status,
                "reason_code": reason_code,
                "questions_asked": len(
                    [event for event in interaction_trace if event["event_type"] == "USER_ANSWER"]
                ),
                "gold_candidate_ids": sorted(gold_candidate_ids),
                "gold_class_ids": sorted(trace_gold_class_ids),
                "gold_class_mapping_unique": len(trace_gold_class_ids) == 1,
                "selected_candidate_ids": list(selected_candidate_ids),
                "gold_candidate_selected": (
                    bool(selected_ids.intersection(gold_candidate_ids))
                    if status == "IDENTIFIED"
                    else False
                ),
            }
        )

    try:
        candidates = _factorize_record(record)
    except CadenceSQLFactorizationError as exc:
        base["reason_code"] = exc.reason_code
        base["factorization_failed"] = True
        append_final_trace(status="ABSTAIN", reason_code=exc.reason_code)
        if memory_runtime is not None:
            base["memory_enabled"] = True
            base["memory_events"] = []
        return base

    trace_gold_class_ids = {
        semantic_class.class_id
        for semantic_class in group_equivalent_candidates(candidates)
        if gold_candidate_ids.intersection(semantic_class.candidate_ids)
    }
    trace_gold_class_id = (
        next(iter(trace_gold_class_ids)) if len(trace_gold_class_ids) == 1 else None
    )
    gold_class_id: str | None = None
    if answer_mode == "gold_oracle":
        if len(trace_gold_class_ids) != 1:
            base["reason_code"] = "CADENCE_GOLD_ORACLE_TARGET_MISSING"
            append_final_trace(
                status="ABSTAIN", reason_code="CADENCE_GOLD_ORACLE_TARGET_MISSING"
            )
            if memory_runtime is not None:
                base["memory_enabled"] = True
                base["memory_events"] = []
            return base
        gold_class_id = trace_gold_class_id
    elif answer_mode != "llm_gold":
        raise ValueError("CADENCE_RUN_ANSWER_MODE_INVALID")

    callbacks = CadenceCallbacks(
        request_text=record.request.question_text,
        gold_class_id=gold_class_id,
        database_schema=record.request.catalog,
        complete=completion,
    )
    predicted_ambiguities: list[str] = []
    oracle_history: list[dict[str, str]] = []
    oracle_decisions: list[dict[str, Any]] = []
    last_attribution: Any | None = None
    last_active_class_ids: tuple[str, ...] = ()

    def attribute(value: Any) -> Any:
        nonlocal last_active_class_ids, last_attribution
        attribution = callbacks.attribute(value)
        last_attribution = attribution
        last_active_class_ids = tuple(value.active_class_ids)
        predicted_ambiguities.append(attribution.ambiguity_text)
        return attribution

    def record_answer_trace(
        question: Any,
        *,
        selected_option_id: str | None,
        decision: Mapping[str, Any],
    ) -> None:
        option_classes: dict[str, tuple[str, ...]] = {}
        if last_attribution is not None:
            option_classes = {
                option.option_id: tuple(option.mapped_node_ids)
                for option in last_attribution.option_mapping.options
            }
        option_ids = sorted(
            question.option_texts,
            key=lambda option_id: option_id.encode("utf-8"),
        )
        gold_option_ids = [
            option_id
            for option_id in option_ids
            if trace_gold_class_id is not None
            and trace_gold_class_id in option_classes.get(option_id, ())
        ]
        options = [
            {
                "option_id": option_id,
                "option_text": question.option_texts[option_id],
                "mapped_class_ids": list(option_classes.get(option_id, ())),
                "contains_gold_class": option_id in gold_option_ids,
            }
            for option_id in option_ids
        ]
        selected_class_ids = (
            option_classes.get(selected_option_id, ())
            if selected_option_id is not None
            else ()
        )
        interaction_trace.append(
            {
                "schema_version": "CadenceInteractionTraceEventV1",
                "event_type": "USER_ANSWER",
                "request_index": record.index,
                "round_index": 1
                + sum(event["event_type"] == "USER_ANSWER" for event in interaction_trace),
                "answer_mode": answer_mode,
                "active_class_ids_before": list(last_active_class_ids),
                "gold_class_id": trace_gold_class_id,
                "gold_class_active_before": (
                    trace_gold_class_id in last_active_class_ids
                    if trace_gold_class_id is not None
                    else False
                ),
                "question_text": question.question_text,
                "options": options,
                "gold_option_ids": gold_option_ids,
                "gold_option_present": len(gold_option_ids) == 1,
                "selected_option_id": selected_option_id,
                "selected_option_is_gold": (
                    selected_option_id in gold_option_ids
                    if selected_option_id is not None
                    else None
                ),
                "gold_class_active_after": (
                    trace_gold_class_id in selected_class_ids
                    if selected_option_id is not None and trace_gold_class_id is not None
                    else None
                ),
                "decision": dict(decision),
            }
        )

    memory_binding = (
        memory_runtime.binding_for(record) if memory_runtime is not None else None
    )
    memory_events: list[dict[str, object]] = []

    def record_feedback(feedback: CadenceFeedbackInput) -> None:
        if memory_runtime is None:
            raise ValueError("CADENCE_MEMORY_FEEDBACK_UNEXPECTED")
        memory_events.append(_memory_event(feedback, record=record, runtime=memory_runtime))

    def answer(question: Any) -> str | None:
        if answer_mode == "gold_oracle":
            try:
                option_id = callbacks.answer(question)
            except Exception as exc:
                reason_code = getattr(exc, "reason_code", "CADENCE_GOLD_ORACLE_FAILED")
                failure_decision = {
                    "decision": "FAILED",
                    "reason_code": str(reason_code),
                }
                oracle_decisions.append(failure_decision)
                record_answer_trace(
                    question,
                    selected_option_id=None,
                    decision=failure_decision,
                )
                raise
            oracle_decision: dict[str, Any] = {
                "decision": "OPTION",
                "answer_mode": "GOLD_CLASS_ORACLE",
                "option_id": option_id,
            }
            oracle_decisions.append(oracle_decision)
            record_answer_trace(
                question,
                selected_option_id=option_id,
                decision=oracle_decision,
            )
            return option_id
        oracle = LLMGoldOptionOracle(completion)
        try:
            option_id, evidence = oracle.choose(
                record.request.question_text,
                record.gold_query,
                question,
                oracle_history,
            )
        except Exception as exc:
            reason_code = getattr(exc, "reason_code", "CADENCE_GOLD_ORACLE_FAILED")
            oracle_decisions.append(
                {"decision": "FAILED", "reason_code": str(reason_code)}
            )
            record_answer_trace(
                question,
                selected_option_id=None,
                decision={"decision": "FAILED", "reason_code": str(reason_code)},
            )
            raise
        oracle_decisions.append(evidence)
        record_answer_trace(
            question,
            selected_option_id=option_id,
            decision=evidence,
        )
        oracle_history.append(
            {
                "clarification_question": question.question_text,
                "selected_option_id": option_id,
                "selected_option_text": question.option_texts[option_id],
            }
        )
        return option_id

    # 每次有效回答至少移除一个活动语义类，因此 C-1 足以覆盖无重复的最坏轮数。
    interaction = run_cadence_interaction(
        candidates=candidates,
        request_text=record.request.question_text,
        attribute=attribute,
        assess_options=callbacks.assess_options,
        generate_question=callbacks.generate_question,
        answer=answer,
        max_questions=max(0, len(candidates) - 1),
        memory=memory_binding,
        record_feedback=record_feedback if memory_runtime is not None else None,
    )
    selected_ids = set(interaction.candidate_ids)
    append_final_trace(
        status=interaction.status,
        reason_code=interaction.reason_code,
        selected_candidate_ids=interaction.candidate_ids,
    )
    final_sql = next(
        (
            candidate_sql[candidate_id]
            for candidate_id in candidate_sql
            if candidate_id in selected_ids
        ),
        None,
    )
    base.update(
        {
            "final_sql": final_sql,
            "predicted_ambiguities": predicted_ambiguities,
            "interaction_status": interaction.status,
            "reason_code": interaction.reason_code,
            "questions_asked": interaction.questions_asked,
            "selected_candidate_ids": list(interaction.candidate_ids),
            "provider_call_count": len(completion.calls),
            "provider_model_ids": sorted(
                {
                    str(call["model_id"])
                    for call in completion.calls
                    if isinstance(call.get("model_id"), str)
                }
            ),
            "memory_enabled": memory_runtime is not None,
            "user_simulator_decisions": oracle_decisions,
        }
    )
    if memory_runtime is not None:
        base["memory_events"] = memory_events
    return base


def _prepare_runtime(
    runtime_config: RuntimeConfig, *, answer_mode: str = "gold_oracle"
) -> dict[str, Any]:
    """检查归因、问句及所选回答模拟器的模型配置和 API key。"""

    roles: dict[str, Any] = {}
    required_roles = set(_CADENCE_RUNTIME_ROLE.values())
    if answer_mode == "gold_oracle":
        required_roles.discard("gold_user_simulation")
    for role in sorted(required_roles):
        resolved = resolve_runtime_role(runtime_config, role)
        require_api_key_env(resolved, os.environ)
        if not resolved.enabled:
            raise ValueError("CADENCE_RUN_RUNTIME_MISMATCH")
        if role != "gold_user_simulation" and (
            resolved.provider_id != "deepseek"
            or resolved.model != "deepseek-flash"
            or resolved.temperature != 0.0
        ):
            raise ValueError("CADENCE_RUN_RUNTIME_MISMATCH")
        roles[role] = {
            "model_id": resolved.model_id,
            "provider_id": resolved.provider_id,
            "model": resolved.model,
            "temperature": resolved.temperature,
            "api_key_env": resolved.api_key_env,
        }
    return roles


def _load_run_records(
    dataset_path: Path,
    database_root: Path,
    repo_root: Path,
) -> tuple[list[Mapping[str, Any]], tuple[ClambsqlRecord, ...]]:
    """用发布数据适配器装载官方候选池并复核 JSON 与请求索引对齐。"""

    dataset_bytes = dataset_path.read_bytes()
    raw = json.loads(dataset_bytes.decode("utf-8"))
    if not isinstance(raw, list) or not all(isinstance(row, Mapping) for row in raw):
        raise ValueError("CADENCE_RUN_DATASET_INVALID")
    dataset = load_clambsql_dataset(
        data_path=dataset_path,
        source_database_root=database_root,
        # request.database_path 不参与候选因子化或评分；外部数据根用明确别名，
        # 最终 SQL EX 始终由 database_root 参数指向的真实 SQLite 文件执行。
        db_snapshot_root=(
            database_root.resolve().relative_to(repo_root).as_posix()
            if database_root.resolve().is_relative_to(repo_root.resolve())
            else f"external/{database_root.resolve().parent.name}"
        ),
        revision=hashlib.sha256(dataset_bytes).hexdigest(),
        engine_name="SQLITE",
    )
    if [int(row["index"]) for row in raw] != [item.index for item in dataset.records]:
        raise ValueError("CADENCE_RUN_DATASET_INDEX_MISMATCH")
    return raw, dataset.records


def _run_manifest(
    *,
    repo_root: Path,
    dataset_path: Path,
    database_root: Path,
    runtime_config_path: Path,
    runtime_roles: Mapping[str, Any],
    records: Sequence[ClambsqlRecord],
    workers: int,
    core_root: Path,
    memory_settings: Mapping[str, Any] | None,
    memory_store_path: Path | None,
    memory_session_start: datetime | None,
    answer_mode: str = "gold_oracle",
) -> dict[str, Any]:
    """固定候选池、源码、数据和模型配置身份，供断点恢复与指标解释使用。"""

    candidate_count = sum(len(record.request.candidates) for record in records)
    memory_enabled = memory_settings is not None and memory_settings.get("enabled") is True
    manifest: dict[str, Any] = {
        "schema_version": _RUN_SCHEMA,
        "method": "CADENCE",
        "candidate_source": "official_CLAMBSQL_ambiguous_queries",
        "request_count": len(records),
        "candidate_slot_count": candidate_count,
        "memory_enabled": memory_enabled,
        "answer_mode": answer_mode,
        "feedback_protocol": (
            "Gold candidate class mapped to its unique option"
            if answer_mode == "gold_oracle"
            else "Gold-conditioned LLM selects a current option"
        ),
        "interaction_trace_path": "trace.jsonl",
        "interaction_trace_schema": "CadenceInteractionTraceEventV1",
        "workers": workers,
        "temperature": 0.0,
        "runtime_roles": dict(runtime_roles),
        "dataset_path": _manifest_path(dataset_path, repo_root),
        "dataset_sha256": sha256_of_file(dataset_path),
        "database_snapshot_id": database_root.resolve().parent.name,
        "database_root": _manifest_path(database_root, repo_root),
        "database_path_contract": (
            "SQL execution uses database_root; StandardRequest path is an external alias"
        ),
        "runtime_config_path": _manifest_path(runtime_config_path, repo_root),
        "runtime_config_sha256": sha256_of_file(runtime_config_path),
        "method_core_source": (
            "cadence_copy" if core_root.resolve() != repo_root.resolve() else "workspace"
        ),
        "method_source_fingerprint": _source_fingerprint(
            core_root, _METHOD_CORE_PATHS
        ),
        "experiment_runner_fingerprint": _source_fingerprint(
            repo_root, _EXPERIMENT_SOURCE_PATHS
        ),
        "scoring": {
            "execution": "typed SQLite result multiset match",
            "ambiguity_schema_entity": "local SQLGlot table/column set diagnostic",
            "ambiguity_annotation_phrase": "normalized exact phrase diagnostic",
            "aggregations": ["micro", "macro"],
        },
    }
    if memory_enabled:
        if (
            memory_settings is None
            or memory_store_path is None
            or memory_session_start is None
        ):
            raise ValueError("CADENCE_MEMORY_CONFIG_INVALID")
        decision = memory_settings.get("decision")
        if not isinstance(decision, Mapping):
            raise ValueError("CADENCE_MEMORY_CONFIG_INVALID")
        manifest.update(
            {
                "memory_store_path": _manifest_path(memory_store_path, repo_root),
                "memory_session_start": (
                    memory_session_start.astimezone(UTC).isoformat().replace("+00:00", "Z")
                ),
                "memory_m_min": decision.get("m_min"),
                "memory_tau_r": (
                    memory_settings.get("retrieval", {}).get("tau_r")
                    if isinstance(memory_settings.get("retrieval"), Mapping)
                    else None
                ),
                "memory_tau_auto": decision.get("tau_auto"),
            }
        )
    return manifest


def run_experiment(
    *,
    dataset_path: Path,
    database_root: Path,
    runtime_config_path: Path,
    output_dir: Path,
    repo_root: Path,
    resume: bool = False,
    workers: int = 4,
    answer_mode: str = "gold_oracle",
) -> dict[str, Any]:
    """并行执行官方候选池；按请求顺序落盘，完整后计算指标。"""

    if workers < 1:
        raise ValueError("CADENCE_RUN_WORKERS_INVALID")

    runtime_config = load_runtime_config(runtime_config_path)
    if answer_mode not in {"gold_oracle", "llm_gold"}:
        raise ValueError("CADENCE_RUN_ANSWER_MODE_INVALID")
    runtime_roles = _prepare_runtime(runtime_config, answer_mode=answer_mode)
    raw_records, records = _load_run_records(dataset_path, database_root, repo_root)
    output_dir = output_dir.resolve()
    memory_settings, memory_store_path = _memory_store_path(
        runtime_config,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    memory_enabled = memory_settings is not None and memory_settings.get("enabled") is True
    if memory_enabled and workers > 1:
        raise ValueError("CADENCE_MEMORY_RUN_REQUIRES_ONE_WORKER")
    manifest_path = output_dir / "run_manifest.json"
    previous_manifest: Mapping[str, Any] | None = None
    if resume:
        if not manifest_path.is_file():
            raise ValueError("CADENCE_RUN_RESUME_MANIFEST_MISSING")
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_manifest, Mapping):
            raise ValueError("CADENCE_RUN_RESUME_FINGERPRINT_MISMATCH")
        previous_manifest = loaded_manifest
    memory_session_start: datetime | None = None
    if memory_enabled:
        if resume:
            session_value = (
                previous_manifest.get("memory_session_start")
                if previous_manifest
                else None
            )
            if not isinstance(session_value, str):
                raise ValueError("CADENCE_MEMORY_SESSION_MISSING")
            try:
                memory_session_start = datetime.fromisoformat(
                    session_value.replace("Z", "+00:00")
                ).astimezone(UTC)
            except ValueError as exc:
                raise ValueError("CADENCE_MEMORY_SESSION_INVALID") from exc
        else:
            memory_session_start = datetime.now(UTC)
    manifest = _run_manifest(
        repo_root=repo_root,
        dataset_path=dataset_path,
        database_root=database_root,
        runtime_config_path=runtime_config_path,
        runtime_roles=runtime_roles,
        records=records,
        workers=workers,
        core_root=_CADENCE_ROOT,
        memory_settings=memory_settings,
        memory_store_path=memory_store_path,
        memory_session_start=memory_session_start,
        answer_mode=answer_mode,
    )
    predictions_path = output_dir / "predictions.jsonl"
    trace_path = output_dir / "trace.jsonl"
    if resume:
        if previous_manifest != manifest:
            raise ValueError("CADENCE_RUN_RESUME_FINGERPRINT_MISMATCH")
    else:
        if output_dir.exists():
            raise ValueError("CADENCE_RUN_OUTPUT_ALREADY_EXISTS")
        output_dir.mkdir(parents=True)
        _atomic_json(manifest_path, manifest)

    completed = _read_completed(predictions_path)
    expected_indices = [record.index for record in records]
    valid_prefix = expected_indices[: len(completed)]
    if list(completed) != valid_prefix:
        raise ValueError("CADENCE_RUN_CHECKPOINT_NOT_CONTIGUOUS")
    memory_runtime = _prepare_memory_runtime(
        memory_settings,
        store_path=memory_store_path,
        session_start=memory_session_start,
    )
    if memory_runtime is not None:
        for row in completed.values():
            events = row.get("memory_events")
            if not isinstance(events, list):
                raise ValueError("CADENCE_MEMORY_CHECKPOINT_INVALID")
            for event in events:
                memory_runtime.store.append(PreferenceEventV1.from_mapping(event))
    start = time.monotonic()
    with (
        predictions_path.open("a", encoding="utf-8", newline="\n") as output,
        trace_path.open("w", encoding="utf-8", newline="\n") as trace_output,
    ):
        for row in completed.values():
            trace_events = row.get("interaction_trace")
            if not isinstance(trace_events, list) or any(
                not isinstance(event, Mapping) for event in trace_events
            ):
                raise ValueError("CADENCE_RUN_TRACE_INVALID")
            for event in trace_events:
                trace_output.write(
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        trace_output.flush()
        remaining = records[len(completed) :]
        next_position = len(completed)
        pending: dict[int, Mapping[str, Any]] = {}

        def execute(record: ClambsqlRecord) -> tuple[int, Mapping[str, Any]]:
            completion = RuntimeCompletion(runtime_config)
            row = run_one_record(
                record,
                completion=completion,
                memory_runtime=memory_runtime,
                answer_mode=answer_mode,
            )
            row["provider_call_count"] = len(completion.calls)
            row["provider_model_ids"] = sorted(
                {
                    str(call["model_id"])
                    for call in completion.calls
                    if isinstance(call.get("model_id"), str)
                }
            )
            return record.index, row

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(execute, record) for record in remaining]
            for future in as_completed(futures):
                index, row = future.result()
                pending[index] = row
                while (
                    next_position < len(records)
                    and records[next_position].index in pending
                ):
                    record = records[next_position]
                    ready_row = pending.pop(record.index)
                    output.write(
                        json.dumps(ready_row, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    output.flush()
                    trace_events = ready_row.get("interaction_trace")
                    if not isinstance(trace_events, list) or any(
                        not isinstance(event, Mapping) for event in trace_events
                    ):
                        raise ValueError("CADENCE_RUN_TRACE_INVALID")
                    for event in trace_events:
                        trace_output.write(
                            json.dumps(
                                event,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    trace_output.flush()
                    completed[record.index] = ready_row
                    if memory_runtime is not None:
                        events = ready_row.get("memory_events")
                        if not isinstance(events, list):
                            raise ValueError("CADENCE_MEMORY_CHECKPOINT_INVALID")
                        for event in events:
                            memory_runtime.store.append(
                                PreferenceEventV1.from_mapping(event)
                            )
                    _atomic_json(
                        output_dir / "progress.json",
                        {
                            "schema_version": "CadenceCLAMBSQLProgress",
                            "completed_requests": len(completed),
                            "request_count": len(records),
                            "elapsed_seconds": time.monotonic() - start,
                            "last_index": record.index,
                        },
                    )
                    next_position += 1

    if len(completed) != len(records):
        raise ValueError("CADENCE_RUN_INCOMPLETE")
    metrics: dict[str, Any] = build_cadence_metrics(
        raw_records,
        completed,
        database_root=database_root,
    )
    metrics["input_hashes"] = {
        "dataset_sha256": manifest["dataset_sha256"],
        "predictions_sha256": sha256_of_file(predictions_path),
        "trace_sha256": sha256_of_file(trace_path),
        "runtime_config_sha256": manifest["runtime_config_sha256"],
        "method_source_fingerprint": manifest["method_source_fingerprint"],
        "experiment_runner_fingerprint": manifest["experiment_runner_fingerprint"],
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; defaults point to the repository's released CLAMBSQL pool."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/clambsql/clambsql.json"))
    parser.add_argument(
        "--database-root",
        type=Path,
        default=Path(
            "data/raw/clambsql/database_snapshot_50645679fdb24803583090d4d6c27ec56269f549/database"
        ),
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("configs/runtime/runtime.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--answer-mode",
        choices=("gold_oracle", "llm_gold"),
        default="gold_oracle",
        help="确定性 Gold 映射或 Gold-conditioned LLM 用户模拟",
    )
    args = parser.parse_args(argv)
    repo_root = _SOURCE_ROOT
    try:
        metrics = run_experiment(
            dataset_path=args.dataset,
            database_root=args.database_root,
            runtime_config_path=args.runtime_config,
            output_dir=args.output_dir,
            repo_root=repo_root,
            resume=args.resume,
            workers=args.workers,
            answer_mode=args.answer_mode,
        )
    except (OSError, ValueError, CadenceSQLFactorizationError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "request_count": metrics["request_count"],
                "execution_match": metrics["execution_match"],
                "schema_entity_metrics": metrics["schema_entity_metrics"],
                "annotation_phrase_metrics": metrics["annotation_phrase_metrics"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
