"""通过统一 JSONL 候选池运行 CADENCE 交互流程。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cadence.candidate_pool import factorize_request
from cadence.datasets.common import (
    DatasetAdapterRejected,
    StandardRequest,
    load_candidate_pool_jsonl,
)
from cadence.factorize_sql import CadenceSQLFactorizationError
from cadence.interaction import (
    CadenceAttribution,
    CadenceAttributionInput,
    CadenceFeedbackInput,
    CadenceMemoryBinding,
    CadenceQuestionDraft,
    CadenceQuestionInput,
    run_cadence_interaction,
)
from cadence.memory import (
    E5_ENCODER_ID,
    MemoryConfig,
    MemoryEngine,
    MemoryScope,
    MultilingualE5SmallEncoder,
    PreferenceEventV1,
    SQLiteMemoryStore,
)
from cadence.provider import call_role
from cadence.runtime_config import (
    RuntimeConfig,
    load_runtime_config,
    require_api_key_env,
    resolve_memory_runtime_config,
    resolve_runtime_role,
)
from tools.cadence.callbacks import CadenceCallbacks, Completion
from tools.cadence.llm_gold_oracle import LLMGoldOptionOracle, LLMGoldOracleError

_CALLBACK_RUNTIME_ROLE = {
    "cadence_attribution": "utterance_attribution",
    "cadence_option_assessment": "utterance_attribution",
    "cadence_question": "clarification_wording",
    "cadence_gold_simulation": "gold_user_simulation",
}


class RuntimeCompletion:
    """把 CADENCE 回调职责映射到配置中的模型角色并记录调用摘要。"""

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        self._runtime_config = runtime_config
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, callback_role: str, messages: Sequence[Mapping[str, str]]
    ) -> str:
        runtime_role = _CALLBACK_RUNTIME_ROLE.get(callback_role)
        if runtime_role is None:
            raise ValueError("CADENCE_CALLBACK_ROLE_UNKNOWN")
        content, metadata = call_role(self._runtime_config, runtime_role, messages)
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
    """统一候选池交互使用的持久偏好记忆会话。"""

    store: SQLiteMemoryStore
    engine: MemoryEngine
    config: MemoryConfig
    m_min: float
    session_id: str
    session_start: datetime

    def binding_for(
        self, request: StandardRequest, *, user_id: str, request_index: int
    ) -> CadenceMemoryBinding:
        scope = MemoryScope(
            namespace="LIVE",
            user_id=user_id,
            database_id=str(Path(request.database_path).resolve()),
            schema_id=_stable_hash(request.catalog),
        )
        return CadenceMemoryBinding(
            engine=self.engine,
            scope=scope,
            config=self.config,
            m_min=self.m_min,
            as_of=self.session_start + timedelta(seconds=request_index),
        )


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _memory_store_path(
    settings: Mapping[str, Any] | None,
    *,
    output_path: Path,
    project_root: Path,
) -> Path | None:
    if settings is None or settings.get("enabled") is False:
        return None
    storage = settings.get("storage")
    if not isinstance(storage, Mapping):
        raise ValueError("CADENCE_MEMORY_CONFIG_INVALID")
    if storage.get("format") != "PreferenceSQLiteVecStoreV1":
        raise ValueError("CADENCE_MEMORY_STORE_FORMAT_UNSUPPORTED")
    configured_path = storage.get("path")
    if configured_path is None:
        return output_path.parent / "memory.sqlite"
    if not isinstance(configured_path, str) or not configured_path.strip():
        raise ValueError("CADENCE_MEMORY_STORE_PATH_INVALID")
    path = Path(configured_path)
    return (path if path.is_absolute() else project_root / path).resolve()


def _prepare_memory_runtime(
    settings: Mapping[str, Any] | None,
    *,
    store_path: Path | None,
    session_start: datetime,
) -> MemoryRuntime | None:
    if settings is None or settings.get("enabled") is False:
        return None
    if store_path is None:
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
    session_start = session_start.astimezone(UTC)
    session_id = _stable_hash(
        {"session_start": session_start.isoformat().replace("+00:00", "Z")}
    )
    return MemoryRuntime(
        store=store,
        engine=engine,
        config=config,
        m_min=float(m_min_raw),
        session_id=session_id,
        session_start=session_start,
    )


def _memory_event(
    feedback: CadenceFeedbackInput,
    *,
    request: StandardRequest,
    runtime: MemoryRuntime,
    user_id: str,
) -> PreferenceEventV1:
    mapping = feedback.attribution.option_mapping
    chosen = next(
        (item for item in mapping.options if item.option_id == feedback.chosen_option_id),
        None,
    )
    if chosen is None or not feedback.candidate_ids:
        raise ValueError("CADENCE_MEMORY_FEEDBACK_INVALID")
    attribution_id = _stable_hash(
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
    option_assessment = feedback.option_assessment
    evidence = json.dumps(
        {
            "semantic_relevance": option_assessment.semantic_relevance,
            "semantic_distinguishability": option_assessment.semantic_distinguishability,
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
        "event_id": _stable_hash(
            {
                "session_id": runtime.session_id,
                "request_id": request.request_id,
                "round_index": feedback.round_index,
                "probe_id": feedback.partition.partition_id,
                "attribution_id": attribution_id,
                "chosen_option_id": chosen.option_id,
            }
        ),
        "namespace": "LIVE",
        "scope": {
            "user_id": user_id,
            "database_id": str(Path(request.database_path).resolve()),
            "schema_id": _stable_hash(request.catalog),
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
            "answer_source": "HUMAN",
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
            "request_id": request.request_id,
            "session_id": runtime.session_id,
            "round_index": feedback.round_index,
            "probe_id": feedback.partition.partition_id,
            "memory_decision_id": (
                feedback.memory_decision.decision_id
                if feedback.memory_decision is not None
                else None
            ),
            "answer_evidence_id": None,
            "attribution_id": attribution_id,
        },
        "occurred_at": occurred_text,
    }
    return PreferenceEventV1.from_mapping(payload)


def _answer_from_terminal(question: CadenceQuestionDraft) -> str | None:
    ordered_options = sorted(
        question.option_texts.items(), key=lambda item: item[0].encode("utf-8")
    )
    print(f"\n{question.question_text}")
    for number, (_, text) in enumerate(ordered_options, start=1):
        print(f"{number}. {text}")
    while True:
        try:
            selected = input("选择序号，或输入 q 弃答: ").strip()
        except EOFError:
            return None
        if not selected or selected.casefold() in {"q", "quit", "abstain"}:
            return None
        if selected in question.option_texts:
            return selected
        if selected.isdecimal():
            index = int(selected)
            if 1 <= index <= len(ordered_options):
                return ordered_options[index - 1][0]
        print("请输入列表序号或选项 ID。")


def _run_request(
    request: StandardRequest,
    *,
    completion: Completion,
    max_questions: int | None,
    memory_runtime: MemoryRuntime | None,
    user_id: str,
    request_index: int,
    answer_mode: str,
) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    ambiguities: list[str] = []
    candidates_by_id = {item["candidate_id"]: item["sql_text"] for item in request.candidates}
    base: dict[str, Any] = {
        "request_id": request.request_id,
        "status": "ABSTAIN",
        "reason_code": None,
        "questions_asked": 0,
        "predicted_ambiguities": ambiguities,
        "questions": questions,
        "selected_candidate_ids": [],
        "selected_candidates": [],
        "provider_call_count": 0,
        "memory_enabled": memory_runtime is not None,
        "answer_mode": answer_mode,
    }
    try:
        candidates = factorize_request(request)
    except (CadenceSQLFactorizationError, ValueError) as exc:
        reason_code = getattr(exc, "reason_code", None)
        base["reason_code"] = (
            reason_code if isinstance(reason_code, str) else str(exc)
        )
        base["factorization_failed"] = True
        return base

    callbacks = CadenceCallbacks(
        request_text=request.question_text,
        database_schema=request.catalog,
        complete=completion,
    )
    oracle_history: list[dict[str, str]] = []
    oracle_decisions: list[dict[str, Any]] = []

    def attribute(value: CadenceAttributionInput) -> CadenceAttribution:
        attribution = callbacks.attribute(value)
        ambiguities.append(attribution.ambiguity_text)
        return attribution

    def generate_question(value: CadenceQuestionInput) -> CadenceQuestionDraft:
        question = callbacks.generate_question(value)
        questions.append(
            {
                "question_text": question.question_text,
                "options": [
                    {"option_id": option_id, "text": text}
                    for option_id, text in sorted(
                        question.option_texts.items(),
                        key=lambda item: item[0].encode("utf-8"),
                    )
                ],
            }
        )
        return question

    memory_binding = (
        memory_runtime.binding_for(
            request, user_id=user_id, request_index=request_index
        )
        if memory_runtime is not None
        else None
    )

    def record_feedback(feedback: CadenceFeedbackInput) -> None:
        if memory_runtime is None:
            raise ValueError("CADENCE_MEMORY_FEEDBACK_UNEXPECTED")
        event = _memory_event(
            feedback,
            request=request,
            runtime=memory_runtime,
            user_id=user_id,
        )
        memory_runtime.store.append(event)

    def answer(question: CadenceQuestionDraft) -> str | None:
        if answer_mode == "terminal":
            return _answer_from_terminal(question)
        if request.gold_sql is None:
            raise LLMGoldOracleError("CADENCE_GOLD_ORACLE_GOLD_SQL_MISSING")
        oracle = LLMGoldOptionOracle(completion)
        try:
            option_id, evidence = oracle.choose(
                request.question_text,
                request.gold_sql,
                question,
                oracle_history,
            )
        except Exception as exc:
            reason_code = getattr(exc, "reason_code", "CADENCE_GOLD_ORACLE_FAILED")
            oracle_decisions.append(
                {"decision": "FAILED", "reason_code": str(reason_code)}
            )
            raise
        oracle_decisions.append(evidence)
        oracle_history.append(
            {
                "clarification_question": question.question_text,
                "selected_option_id": option_id,
                "selected_option_text": question.option_texts[option_id],
            }
        )
        return option_id

    interaction = run_cadence_interaction(
        candidates=candidates,
        request_text=request.question_text,
        attribute=attribute,
        assess_options=callbacks.assess_options,
        generate_question=generate_question,
        answer=answer,
        max_questions=(
            max(0, len(candidates) - 1)
            if max_questions is None
            else max_questions
        ),
        memory=memory_binding,
        record_feedback=record_feedback if memory_runtime is not None else None,
    )
    selected_ids = set(interaction.candidate_ids)
    selected_candidates = [
        {"candidate_id": candidate_id, "sql_text": candidates_by_id[candidate_id]}
        for candidate_id in candidates_by_id
        if candidate_id in selected_ids
    ]
    base.update(
        {
            "status": interaction.status,
            "reason_code": interaction.reason_code,
            "questions_asked": interaction.questions_asked,
            "selected_candidate_ids": list(interaction.candidate_ids),
            "selected_candidates": selected_candidates,
            "provider_call_count": len(getattr(completion, "calls", [])),
            "user_simulator_decisions": oracle_decisions,
        }
    )
    return base


def _prepare_runtime(runtime_config: RuntimeConfig, *, answer_mode: str) -> None:
    required_roles = set(_CALLBACK_RUNTIME_ROLE.values())
    if answer_mode == "terminal":
        required_roles.discard("gold_user_simulation")
    for role in sorted(required_roles):
        resolved = resolve_runtime_role(runtime_config, role)
        require_api_key_env(resolved, os.environ)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("configs/runtime/default.yaml"),
    )
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--user-id", default="local-user")
    parser.add_argument(
        "--answer-mode",
        choices=("terminal", "llm_gold"),
        default="terminal",
        help="选择终端用户回答或 Gold-conditioned LLM 用户模拟",
    )
    args = parser.parse_args(argv)
    if args.max_questions is not None and args.max_questions < 0:
        parser.error("--max-questions must be non-negative")

    project_root = Path(__file__).resolve().parents[2]
    output_path = args.output.resolve()
    if output_path.exists():
        parser.error("CADENCE_STANDARD_RUN_OUTPUT_EXISTS")
    try:
        requests = load_candidate_pool_jsonl(args.candidate_pool)
        if args.answer_mode == "llm_gold" and any(
            request.gold_sql is None for request in requests
        ):
            raise ValueError("CADENCE_GOLD_ORACLE_GOLD_SQL_REQUIRED")
        runtime_config = load_runtime_config(args.runtime_config)
        _prepare_runtime(runtime_config, answer_mode=args.answer_mode)
        memory_settings = resolve_memory_runtime_config(runtime_config)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        memory_runtime = _prepare_memory_runtime(
            memory_settings,
            store_path=_memory_store_path(
                memory_settings,
                output_path=output_path,
                project_root=project_root,
            ),
            session_start=datetime.now(UTC),
        )
        with output_path.open("x", encoding="utf-8", newline="\n") as output:
            for request_index, request in enumerate(requests):
                completion = RuntimeCompletion(runtime_config)
                result = _run_request(
                    request,
                    completion=completion,
                    max_questions=args.max_questions,
                    memory_runtime=memory_runtime,
                    user_id=args.user_id,
                    request_index=request_index,
                    answer_mode=args.answer_mode,
                )
                result["provider_call_count"] = len(completion.calls)
                output.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                output.flush()
                print(f"{request.request_id}: {result['status']}")
    except (DatasetAdapterRejected, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
