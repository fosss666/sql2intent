"""偏好记忆事件、参数、范围与动作类型。

本模块是独立记忆核心的唯一类型来源。事件保存显式反馈证据；缺字段、未知字段、
无效准入或候选映射不唯一时拒绝，不猜测缺失含义。
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RFC3339_UTC = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)
_ENGINE_WITH_HISTORY_FLOOR_PARAMETER_SET_ID = (
    "CADENCE_MEMORY_ENGINE_WITH_HISTORY_FLOOR"
)
_ENGINE_NO_HISTORY_FLOOR_PARAMETER_SET_ID = "CADENCE_MEMORY_ENGINE_NO_HISTORY_FLOOR"
_FIXED_SEMANTIC_TOP_K = 20
_NO_HISTORY_FLOOR_PARAMETERS = (0.75, 30.0, None, 0.85, _FIXED_SEMANTIC_TOP_K)
_HISTORY_FLOOR_PARAMETERS = (0.75, 30.0, 2.0, 0.85, _FIXED_SEMANTIC_TOP_K)
_FIXED_MIN_EFFECTIVE_HISTORY = 2.0
_EVENT_KEYS = {
    "schema_version",
    "event_id",
    "namespace",
    "scope",
    "retrieval_text",
    "preference",
    "feedback_action",
    "admission",
    "mapping_snapshot",
    "source_refs",
    "occurred_at",
}


class PreferenceMemoryError(ValueError):
    """记忆组件稳定失败；`reason_code` 可用于调用方机器处理。"""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(reason_code, detail)
        self.reason_code = reason_code
        self.detail = detail


class MemoryDecisionAction(StrEnum):
    """记忆决策支持的三种动作。"""

    ASK = "ASK"
    RECOMMEND = "RECOMMEND"
    AUTO_SELECT = "AUTO_SELECT"


class TextEncoder(Protocol):
    """配置绑定的文本向量编码器接口；具体模型由运行配置显式指定。"""

    encoder_id: str

    def encode(self, text: str) -> Sequence[float]:
        """将输入文本编码为有限维向量；同一 encoder 必须确定性输出。"""


def _require_closed_mapping(
    value: object,
    expected: set[str],
    *,
    code: str = "MEMORY_EVENT_SCHEMA_INVALID",
    field: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PreferenceMemoryError(code, field)
    return value


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", field)
    return value


def _require_hex64(value: object, field: str) -> str:
    result = _require_nonempty_string(value, field)
    if not _HEX64.fullmatch(result):
        raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", field)
    return result


def _parse_utc(value: object, field: str) -> datetime:
    raw = _require_nonempty_string(value, field)
    if not _RFC3339_UTC.fullmatch(raw):
        raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", field)
    try:
        result = datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as exc:
        raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", field) from exc
    if result.utcoffset() != UTC.utcoffset(result):
        raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", field)
    return result


@dataclass(frozen=True)
class MemoryScope:
    """标识偏好事件的命名空间、用户、数据库和 schema。"""

    namespace: str
    user_id: str
    database_id: str
    schema_id: str

    def __post_init__(self) -> None:
        if not (
            self.namespace == "LIVE"
            or (self.namespace.startswith("EXPERIMENT:") and self.namespace[11:])
        ):
            raise PreferenceMemoryError("MEMORY_SCOPE_INVALID", "namespace")
        for name in ("user_id", "database_id"):
            if not getattr(self, name).strip():
                raise PreferenceMemoryError("MEMORY_SCOPE_INVALID", name)
        if not _HEX64.fullmatch(self.schema_id):
            raise PreferenceMemoryError("MEMORY_SCOPE_INVALID", "schema_id")


@dataclass(frozen=True)
class CandidateOption:
    """本轮冻结 option 到稳定偏好值及节点的单值映射。"""

    option_id: str
    preference_value_id: str
    mapped_node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.option_id.strip() or not self.preference_value_id.strip():
            raise PreferenceMemoryError("MEMORY_CANDIDATE_MAPPING_INVALID")
        if not self.mapped_node_ids or any(not item.strip() for item in self.mapped_node_ids):
            raise PreferenceMemoryError("MEMORY_CANDIDATE_MAPPING_INVALID")
        if len(set(self.mapped_node_ids)) != len(self.mapped_node_ids):
            raise PreferenceMemoryError("MEMORY_CANDIDATE_MAPPING_INVALID")


@dataclass(frozen=True)
class CandidateOptionMapping:
    """冻结候选集的稳定偏好值映射；同一偏好值不得落到多个 option。"""

    candidate_set_key: str
    options: tuple[CandidateOption, ...]

    def __post_init__(self) -> None:
        if not _HEX64.fullmatch(self.candidate_set_key) or len(self.options) < 2:
            raise PreferenceMemoryError("MEMORY_CANDIDATE_MAPPING_INVALID")
        option_ids = [item.option_id for item in self.options]
        value_ids = [item.preference_value_id for item in self.options]
        if len(set(option_ids)) != len(option_ids) or len(set(value_ids)) != len(value_ids):
            raise PreferenceMemoryError("MEMORY_CANDIDATE_MAPPING_INVALID")

    def option_for_preference(self, preference_value_id: str) -> CandidateOption | None:
        """只返回唯一、当前仍存在的映射；不将旧值重分配给近似候选。"""

        matches = [
            option for option in self.options if option.preference_value_id == preference_value_id
        ]
        return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class MemoryConfig:
    """显式配置 encoder、top-k、时间衰减与选择阈值。

    ``min_effective_history`` 保留给通用调用方；CADENCE 将该值设为 ``None``，
    并在方法策略中单独应用 ``M_min``。
    """

    encoder_id: str
    tau_retrieval: float
    half_life_days: float
    min_effective_history: float | None
    tau_auto: float
    semantic_top_k: int = _FIXED_SEMANTIC_TOP_K
    # 参数集标识参与 decision ID，避免不同阈值配置产生相同的决策身份。
    parameter_set_id: str = _ENGINE_WITH_HISTORY_FLOOR_PARAMETER_SET_ID
    calibration_id: str | None = None
    validation_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.encoder_id.strip() or not self.parameter_set_id.strip():
            raise PreferenceMemoryError("MEMORY_CONFIG_INVALID")
        if self.semantic_top_k != _FIXED_SEMANTIC_TOP_K:
            raise PreferenceMemoryError("MEMORY_CONFIG_INVALID", "semantic_top_k")
        if not 0.0 <= self.tau_retrieval <= 1.0:
            raise PreferenceMemoryError("MEMORY_CONFIG_INVALID", "tau_retrieval")
        numeric_values: tuple[float, ...] = (
            self.tau_retrieval,
            self.half_life_days,
            self.tau_auto,
            float(self.semantic_top_k),
        )
        if self.min_effective_history is not None:
            numeric_values += (self.min_effective_history,)
        if any(not math.isfinite(value) for value in numeric_values):
            raise PreferenceMemoryError("MEMORY_CONFIG_INVALID", "non-finite parameter")
        if self.half_life_days <= 0 or (
            self.min_effective_history is not None
            and self.min_effective_history < 0
        ):
            raise PreferenceMemoryError("MEMORY_CONFIG_INVALID", "scoring")
        if self.min_effective_history not in (None, _FIXED_MIN_EFFECTIVE_HISTORY):
            raise PreferenceMemoryError("MEMORY_CONFIG_INVALID", "min_effective_history")
        if not 0.0 <= self.tau_auto <= 1.0:
            raise PreferenceMemoryError("MEMORY_CONFIG_INVALID", "tau_auto")
        values = (
            self.tau_retrieval,
            self.half_life_days,
            self.min_effective_history,
            self.tau_auto,
            self.semantic_top_k,
        )
        if self.parameter_set_id == _ENGINE_NO_HISTORY_FLOOR_PARAMETER_SET_ID:
            if values != _NO_HISTORY_FLOOR_PARAMETERS or any(
                item is not None for item in (self.calibration_id, self.validation_fingerprint)
            ):
                raise PreferenceMemoryError(
                    "MEMORY_CONFIG_INVALID", "memory parameter set mismatch"
                )
        elif self.parameter_set_id == _ENGINE_WITH_HISTORY_FLOOR_PARAMETER_SET_ID:
            if values != _HISTORY_FLOOR_PARAMETERS or any(
                item is not None for item in (self.calibration_id, self.validation_fingerprint)
            ):
                raise PreferenceMemoryError(
                    "MEMORY_CONFIG_INVALID", "memory parameter set mismatch"
                )
        elif (
            not isinstance(self.calibration_id, str)
            or not self.calibration_id.strip()
            or not isinstance(self.validation_fingerprint, str)
            or not _HEX64.fullmatch(self.validation_fingerprint)
        ):
            raise PreferenceMemoryError("MEMORY_CONFIG_INVALID", "calibration identity required")


@dataclass(frozen=True)
class PreferenceEventV1:
    """严格验证后的 `PreferenceEventV1`；原始映射是追加账本中的事实内容。"""

    event_id: str
    namespace: str
    scope: MemoryScope
    ambiguity_key: str
    ambiguity_text: str
    context_key: str | None
    context_text: str | None
    preference_value_id: str
    preference_value_text: str
    feedback_action: str
    user_distinguishability: str
    assessment_validation_status: str
    assessment_evidence: str
    answer_source: str
    answer_kind: str
    grounding_status: str
    reliably_committed: bool
    candidate_set_key: str
    candidate_ids: tuple[str, ...]
    option_mappings: tuple[CandidateOption, ...]
    chosen_option_id: str
    recommended_option_id: str | None
    recommended_preference_value_id: str | None
    mapped_node_ids: tuple[str, ...]
    request_id: str
    session_id: str | None
    round_index: int
    probe_id: str
    memory_decision_id: str | None
    answer_evidence_id: str | None
    attribution_id: str
    occurred_at: datetime

    @classmethod
    def from_mapping(cls, value: object) -> PreferenceEventV1:
        """验证封闭事件和准入条件；不接受结构上未知或不可追溯的字段。"""

        event = _require_closed_mapping(value, _EVENT_KEYS, field="event")
        if event["schema_version"] != "PreferenceEventV1":
            raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", "schema_version")
        event_id = _require_hex64(event["event_id"], "event_id")
        namespace = _require_nonempty_string(event["namespace"], "namespace")
        scope_map = _require_closed_mapping(
            event["scope"], {"user_id", "database_id", "schema_id"}, field="scope"
        )
        scope = MemoryScope(
            namespace=namespace,
            user_id=_require_nonempty_string(scope_map["user_id"], "scope.user_id"),
            database_id=_require_nonempty_string(scope_map["database_id"], "scope.database_id"),
            schema_id=_require_hex64(scope_map["schema_id"], "scope.schema_id"),
        )
        retrieval = _require_closed_mapping(
            event["retrieval_text"],
            {"ambiguity_key", "ambiguity_text", "context_key", "context_text"},
            field="retrieval_text",
        )
        ambiguity_key = _require_nonempty_string(
            retrieval["ambiguity_key"], "retrieval_text.ambiguity_key"
        )
        ambiguity_text = _require_nonempty_string(
            retrieval["ambiguity_text"], "retrieval_text.ambiguity_text"
        )
        context_key = retrieval["context_key"]
        context_text = retrieval["context_text"]
        if context_key is not None:
            context_key = _require_nonempty_string(context_key, "retrieval_text.context_key")
        if context_text is not None:
            context_text = _require_nonempty_string(context_text, "retrieval_text.context_text")
        preference = _require_closed_mapping(
            event["preference"],
            {"preference_schema_version", "preference_value_id", "preference_value_text"},
            field="preference",
        )
        if preference["preference_schema_version"] != "PreferenceValueV1":
            raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", "preference_schema_version")
        preference_value_id = _require_nonempty_string(
            preference["preference_value_id"], "preference_value_id"
        )
        preference_value_text = _require_nonempty_string(
            preference["preference_value_text"], "preference_value_text"
        )
        feedback_action = _require_nonempty_string(event["feedback_action"], "feedback_action")
        admission = _require_closed_mapping(
            event["admission"],
            {
                "memory_admission_assessment",
                "answer_source",
                "answer_kind",
                "grounding_status",
                "reliably_committed",
            },
            field="admission",
        )
        assessment = _require_closed_mapping(
            admission["memory_admission_assessment"],
            {"user_distinguishability", "validation_status", "evidence"},
            code="MEMORY_FEEDBACK_NOT_ADMISSIBLE",
            field="admission.memory_admission_assessment",
        )
        distinguishability = _require_nonempty_string(
            assessment["user_distinguishability"],
            "memory_admission_assessment.user_distinguishability",
        )
        assessment_validation_status = _require_nonempty_string(
            assessment["validation_status"],
            "memory_admission_assessment.validation_status",
        )
        assessment_evidence_raw = assessment["evidence"]
        if not isinstance(assessment_evidence_raw, str) or not assessment_evidence_raw.strip():
            raise PreferenceMemoryError("MEMORY_FEEDBACK_NOT_ADMISSIBLE")
        assessment_evidence = assessment_evidence_raw
        answer_source = _require_nonempty_string(admission["answer_source"], "answer_source")
        answer_kind = _require_nonempty_string(admission["answer_kind"], "answer_kind")
        grounding_status = _require_nonempty_string(
            admission["grounding_status"], "grounding_status"
        )
        reliably_committed = admission["reliably_committed"]
        if not isinstance(reliably_committed, bool):
            raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", "reliably_committed")
        if (
            distinguishability != "DISTINGUISHABLE"
            or assessment_validation_status != "ASSESSED"
            or answer_kind != "OPTION"
            or grounding_status != "GROUNDED_OPTION"
            or not reliably_committed
            or answer_source not in {"HUMAN", "ORACLE_SIMULATED", "REPLAY"}
            or (namespace == "LIVE" and answer_source != "HUMAN")
            or (namespace != "LIVE" and answer_source == "HUMAN")
        ):
            raise PreferenceMemoryError("MEMORY_FEEDBACK_NOT_ADMISSIBLE")
        mapping = _require_closed_mapping(
            event["mapping_snapshot"],
            {
                "candidate_set_key",
                "candidate_ids",
                "option_mappings",
                "chosen_option_id",
                "recommended_option_id",
                "recommended_preference_value_id",
                "mapped_node_ids",
            },
            field="mapping_snapshot",
        )
        candidate_set_key = _require_hex64(
            mapping["candidate_set_key"], "mapping_snapshot.candidate_set_key"
        )
        candidate_ids = _string_tuple(mapping["candidate_ids"], "candidate_ids")
        raw_option_mappings = mapping["option_mappings"]
        if not isinstance(raw_option_mappings, list) or len(raw_option_mappings) < 2:
            raise PreferenceMemoryError("MEMORY_EVENT_MAPPING_INVALID", "option_mappings")
        option_mappings: list[CandidateOption] = []
        for raw_option in raw_option_mappings:
            option_map = _require_closed_mapping(
                raw_option,
                {"option_id", "preference_value_id", "mapped_node_ids"},
                code="MEMORY_EVENT_MAPPING_INVALID",
                field="mapping_snapshot.option_mappings",
            )
            option_mappings.append(
                CandidateOption(
                    option_id=_require_nonempty_string(
                        option_map["option_id"], "option_mappings.option_id"
                    ),
                    preference_value_id=_require_nonempty_string(
                        option_map["preference_value_id"],
                        "option_mappings.preference_value_id",
                    ),
                    mapped_node_ids=_string_tuple(
                        option_map["mapped_node_ids"],
                        "option_mappings.mapped_node_ids",
                    ),
                )
            )
        option_ids = [item.option_id for item in option_mappings]
        option_value_ids = [item.preference_value_id for item in option_mappings]
        mapped_nodes = [node_id for item in option_mappings for node_id in item.mapped_node_ids]
        if (
            len(set(option_ids)) != len(option_ids)
            or len(set(option_value_ids)) != len(option_value_ids)
            or len(set(mapped_nodes)) != len(mapped_nodes)
        ):
            raise PreferenceMemoryError(
                "MEMORY_EVENT_MAPPING_INVALID", "option_mappings not one-to-one"
            )
        chosen_option_id = _require_nonempty_string(mapping["chosen_option_id"], "chosen_option_id")
        recommended_option_id = mapping["recommended_option_id"]
        recommended_preference_value_id = mapping["recommended_preference_value_id"]
        if recommended_option_id is not None:
            recommended_option_id = _require_nonempty_string(
                recommended_option_id, "recommended_option_id"
            )
        if recommended_preference_value_id is not None:
            recommended_preference_value_id = _require_nonempty_string(
                recommended_preference_value_id, "recommended_preference_value_id"
            )
        mapped_node_ids = _string_tuple(mapping["mapped_node_ids"], "mapped_node_ids")
        chosen_mapping = next(
            (item for item in option_mappings if item.option_id == chosen_option_id),
            None,
        )
        if (
            chosen_mapping is None
            or chosen_mapping.preference_value_id != preference_value_id
            or chosen_mapping.mapped_node_ids != mapped_node_ids
        ):
            raise PreferenceMemoryError(
                "MEMORY_EVENT_MAPPING_INVALID", "chosen option mapping mismatch"
            )
        source_refs = _require_closed_mapping(
            event["source_refs"],
            {
                "request_id",
                "session_id",
                "round_index",
                "probe_id",
                "memory_decision_id",
                "answer_evidence_id",
                "attribution_id",
            },
            field="source_refs",
        )
        request_id = _require_nonempty_string(source_refs["request_id"], "request_id")
        session_id = _optional_string(source_refs["session_id"], "session_id")
        round_index = source_refs["round_index"]
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
            raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", "round_index")
        probe_id = _require_nonempty_string(source_refs["probe_id"], "probe_id")
        memory_decision_id = _optional_string(
            source_refs["memory_decision_id"], "memory_decision_id"
        )
        answer_evidence_id = _optional_string(
            source_refs["answer_evidence_id"], "answer_evidence_id"
        )
        attribution_id = _require_hex64(source_refs["attribution_id"], "attribution_id")
        if feedback_action == "ASK_OPTION_SELECTED":
            if recommended_option_id is not None or recommended_preference_value_id is not None:
                raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", "ASK recommendation")
        elif feedback_action in {"RECOMMENDATION_CONFIRMED", "RECOMMENDATION_OVERRIDDEN"}:
            if (
                not memory_decision_id
                or not recommended_option_id
                or not recommended_preference_value_id
            ):
                raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", "recommendation refs")
            recommended_mapping = next(
                (item for item in option_mappings if item.option_id == recommended_option_id),
                None,
            )
            if (
                recommended_mapping is None
                or recommended_mapping.preference_value_id != recommended_preference_value_id
            ):
                raise PreferenceMemoryError(
                    "MEMORY_EVENT_MAPPING_INVALID", "recommended option mapping mismatch"
                )
            if feedback_action == "RECOMMENDATION_CONFIRMED":
                if chosen_option_id != recommended_option_id:
                    raise PreferenceMemoryError(
                        "MEMORY_EVENT_MAPPING_INVALID", "confirmation option mismatch"
                    )
            elif chosen_option_id == recommended_option_id:
                raise PreferenceMemoryError(
                    "MEMORY_EVENT_MAPPING_INVALID", "override option mismatch"
                )
        else:
            raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", "feedback_action")
        return cls(
            event_id=event_id,
            namespace=namespace,
            scope=scope,
            ambiguity_key=ambiguity_key,
            ambiguity_text=ambiguity_text,
            context_key=context_key,
            context_text=context_text,
            preference_value_id=preference_value_id,
            preference_value_text=preference_value_text,
            feedback_action=feedback_action,
            user_distinguishability=distinguishability,
            assessment_validation_status=assessment_validation_status,
            assessment_evidence=assessment_evidence,
            answer_source=answer_source,
            answer_kind=answer_kind,
            grounding_status=grounding_status,
            reliably_committed=reliably_committed,
            candidate_set_key=candidate_set_key,
            candidate_ids=candidate_ids,
            option_mappings=tuple(option_mappings),
            chosen_option_id=chosen_option_id,
            recommended_option_id=recommended_option_id,
            recommended_preference_value_id=recommended_preference_value_id,
            mapped_node_ids=mapped_node_ids,
            request_id=request_id,
            session_id=session_id,
            round_index=round_index,
            probe_id=probe_id,
            memory_decision_id=memory_decision_id,
            answer_evidence_id=answer_evidence_id,
            attribution_id=attribution_id,
            occurred_at=_parse_utc(event["occurred_at"], "occurred_at"),
        )

    @property
    def scope_key(self) -> tuple[str, str, str, str]:
        """事件来源键；含 user_id 供追溯，不等同于跨用户共享的召回 scope。"""

        return (
            self.namespace,
            self.scope.user_id,
            self.scope.database_id,
            self.scope.schema_id,
        )

    def to_mapping(self) -> dict[str, object]:
        """返回可写入 JSONL 的封闭结构，并统一时间为 UTC `Z`。"""

        if (
            not isinstance(self.occurred_at, datetime)
            or self.occurred_at.tzinfo is None
            or self.occurred_at.utcoffset() is None
        ):
            # naive datetime 的 astimezone 会隐式读取机器本地时区，可能跨 UTC 日期，
            # 从而改变日历日半衰期；拒绝序列化以保持事件时间可重放。
            raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", "occurred_at")

        return {
            "schema_version": "PreferenceEventV1",
            "event_id": self.event_id,
            "namespace": self.namespace,
            "scope": {
                "user_id": self.scope.user_id,
                "database_id": self.scope.database_id,
                "schema_id": self.scope.schema_id,
            },
            "retrieval_text": {
                "ambiguity_key": self.ambiguity_key,
                "ambiguity_text": self.ambiguity_text,
                "context_key": self.context_key,
                "context_text": self.context_text,
            },
            "preference": {
                "preference_schema_version": "PreferenceValueV1",
                "preference_value_id": self.preference_value_id,
                "preference_value_text": self.preference_value_text,
            },
            "feedback_action": self.feedback_action,
            "admission": {
                "memory_admission_assessment": {
                    "user_distinguishability": self.user_distinguishability,
                    "validation_status": self.assessment_validation_status,
                    "evidence": self.assessment_evidence,
                },
                "answer_source": self.answer_source,
                "answer_kind": self.answer_kind,
                "grounding_status": self.grounding_status,
                "reliably_committed": self.reliably_committed,
            },
            "mapping_snapshot": {
                "candidate_set_key": self.candidate_set_key,
                "candidate_ids": list(self.candidate_ids),
                "option_mappings": [
                    {
                        "option_id": item.option_id,
                        "preference_value_id": item.preference_value_id,
                        "mapped_node_ids": list(item.mapped_node_ids),
                    }
                    for item in self.option_mappings
                ],
                "chosen_option_id": self.chosen_option_id,
                "recommended_option_id": self.recommended_option_id,
                "recommended_preference_value_id": self.recommended_preference_value_id,
                "mapped_node_ids": list(self.mapped_node_ids),
            },
            "source_refs": {
                "request_id": self.request_id,
                "session_id": self.session_id,
                "round_index": self.round_index,
                "probe_id": self.probe_id,
                "memory_decision_id": self.memory_decision_id,
                "answer_evidence_id": self.answer_evidence_id,
                "attribution_id": self.attribution_id,
            },
            "occurred_at": self.occurred_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        }


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, field)


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", field)
    items = tuple(_require_nonempty_string(item, field) for item in value)
    if len(set(items)) != len(items):
        raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", field)
    return items


@dataclass(frozen=True)
class MemoryEvidence:
    """确定性评分证据；首选占比不解释为当前意图概率。"""

    support_by_value: tuple[tuple[str, float], ...]
    effective_history: float
    top_share: float
    preferred_value_id: str
    retrieved_event_ids: tuple[str, ...]
    retrieval_mode: str


@dataclass(frozen=True)
class MemoryDecision:
    """决策器结果；`ASK` 时不携带选项，避免调用方误用未知偏好。"""

    action: MemoryDecisionAction
    reason_code: str
    decision_id: str
    preference_value_id: str | None = None
    option_id: str | None = None
    evidence: MemoryEvidence | None = None
    retrieval_mode: str = "NONE"
