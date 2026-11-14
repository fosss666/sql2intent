"""将记忆检索结果按 CADENCE 的 ASK/AUTO_SELECT 规则收口。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime

from cadence.constants import (
    CADENCE_COMPATIBLE_BLOCKED_BY_OTHER_SUPPORTED,
    CADENCE_FACTOR_ORDER,
    CADENCE_INTENT_ASSESSMENT_LABELS,
    CADENCE_INTENT_BLOCKING_STATUSES,
    CADENCE_INTENT_COMPATIBLE_STATUS,
    CADENCE_INTENT_SUPPORTED_STATUS,
    CADENCE_MEMORY_POLICY_ID,
)
from cadence.memory import (
    CandidateOptionMapping,
    MemoryConfig,
    MemoryDecision,
    MemoryDecisionAction,
    MemoryEngine,
    MemoryScope,
)

CADENCE_MEMORY_AUTO_THRESHOLD_NOT_MET = "CADENCE_MEMORY_AUTO_THRESHOLD_NOT_MET"
CADENCE_MEMORY_INSUFFICIENT_SUPPORT = "CADENCE_MEMORY_INSUFFICIENT_SUPPORT"
CADENCE_MEMORY_OPTION_DOES_NOT_NARROW = "CADENCE_MEMORY_OPTION_DOES_NOT_NARROW"
CADENCE_MEMORY_CONFIG_INVALID = "CADENCE_MEMORY_CONFIG_INVALID"
CADENCE_MEMORY_CURRENT_INTENT_ASSESSMENT_INVALID = (
    "CADENCE_MEMORY_CURRENT_INTENT_ASSESSMENT_INVALID"
)
CADENCE_MEMORY_CURRENT_INTENT_CONFLICT = "CADENCE_MEMORY_CURRENT_INTENT_CONFLICT"
CADENCE_MEMORY_ROOT_TYPE_INVALID = "CADENCE_MEMORY_ROOT_TYPE_INVALID"


class CadenceMemoryPolicyError(ValueError):
    """Memory 配置无效；reason_code 可供实验入口稳定处理。"""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _decision_for_policy(
    decision: MemoryDecision,
    *,
    policy_id: str,
    m_min: float,
    action: MemoryDecisionAction,
    reason_code: str,
    option_intent_assessments: Mapping[str, str] | None = None,
) -> MemoryDecision:
    """把检索结果绑定到 CADENCE 动作集，生成稳定的决策 ID。"""

    preimage = {
        "policy_id": policy_id,
        "m_min": float(m_min),
        "base_decision_id": decision.decision_id,
        "action": action.value,
        "reason_code": reason_code,
        "option_intent_assessments": (
            dict(sorted(option_intent_assessments.items()))
            if option_intent_assessments is not None
            and all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in option_intent_assessments.items()
            )
            else None
        ),
    }
    encoded = json.dumps(
        preimage,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return replace(
        decision,
        action=action,
        reason_code=reason_code,
        decision_id=hashlib.sha256(encoded).hexdigest(),
        preference_value_id=(
            decision.preference_value_id
            if action is MemoryDecisionAction.AUTO_SELECT
            else None
        ),
        option_id=(decision.option_id if action is MemoryDecisionAction.AUTO_SELECT else None),
    )


def _active_partition_is_valid(
    mapping: CandidateOptionMapping,
    active_node_ids: Sequence[str],
) -> bool:
    """确认冻结 options 是活动候选类集合的完整、互斥分区。"""

    if (
        isinstance(active_node_ids, (str, bytes))
        or not isinstance(active_node_ids, Sequence)
        or not active_node_ids
        or any(not isinstance(item, str) or not item.strip() for item in active_node_ids)
        or len(set(active_node_ids)) != len(active_node_ids)
    ):
        return False
    active = set(active_node_ids)
    covered: set[str] = set()
    for option in mapping.options:
        option_nodes = set(option.mapped_node_ids)
        if covered.intersection(option_nodes):
            return False
        covered.update(option_nodes)
    return covered == active


def decide_cadence_memory(
    engine: MemoryEngine,
    *,
    scope: MemoryScope,
    query_ambiguity_key: str,
    query_ambiguity_text: str,
    query_context_key: str | None,
    query_context_text: str | None,
    mapping: CandidateOptionMapping,
    config: MemoryConfig,
    m_min: float,
    active_node_ids: Sequence[str],
    option_intent_assessments: Mapping[str, str],
    as_of: datetime,
) -> MemoryDecision:
    """应用 CADENCE Memory 条件，返回 ASK 或 AUTO_SELECT。

    ``query_ambiguity_key`` 是当前被选择的 semantic root type；历史同根类型记录再按
    归因出的自然语言证据、完整规范化选项集和时间衰减进行检索。检索、匹配和 preference
    score 完全来自独立 MemoryEngine。若任一
    自动选择条件不满足时输出 ASK 且不携带 preference/option，确保调用方不能把
    历史首选误当成建议。AUTO_SELECT 的 options 必须覆盖且互斥地划分当前活动视图。
    """

    if (
        isinstance(m_min, bool)
        or not isinstance(m_min, (int, float))
        or not math.isfinite(float(m_min))
        or m_min < 0
        or config.min_effective_history is not None
    ):
        raise CadenceMemoryPolicyError(CADENCE_MEMORY_CONFIG_INVALID)

    if (
        not isinstance(query_ambiguity_key, str)
        or query_ambiguity_key not in CADENCE_FACTOR_ORDER
    ):
        raise CadenceMemoryPolicyError(CADENCE_MEMORY_ROOT_TYPE_INVALID)
    intent_labels = frozenset(CADENCE_INTENT_ASSESSMENT_LABELS)
    decision = engine.decide(
        scope=scope,
        query_ambiguity_key=query_ambiguity_key,
        query_ambiguity_text=query_ambiguity_text,
        query_context_key=query_context_key,
        query_context_text=query_context_text,
        mapping=mapping,
        config=config,
        as_of=as_of,
        require_cadence_option_set=True,
    )

    def finish(action: MemoryDecisionAction, reason_code: str) -> MemoryDecision:
        """生成绑定当前策略和意图评估的稳定 decision ID。"""

        return _decision_for_policy(
            decision,
            policy_id=CADENCE_MEMORY_POLICY_ID,
            m_min=float(m_min),
            action=action,
            reason_code=reason_code,
            option_intent_assessments=option_intent_assessments,
        )

    if decision.action is MemoryDecisionAction.ASK:
        return finish(MemoryDecisionAction.ASK, decision.reason_code)
    if decision.action is MemoryDecisionAction.RECOMMEND:
        return finish(MemoryDecisionAction.ASK, CADENCE_MEMORY_AUTO_THRESHOLD_NOT_MET)

    evidence = decision.evidence
    if evidence is None:
        return finish(MemoryDecisionAction.ASK, CADENCE_MEMORY_AUTO_THRESHOLD_NOT_MET)
    if evidence.effective_history < m_min:
        return finish(MemoryDecisionAction.ASK, CADENCE_MEMORY_INSUFFICIENT_SUPPORT)
    supports = tuple(value for _preference_id, value in evidence.support_by_value)
    if (
        evidence.top_share < config.tau_auto
        or not supports
        or sum(value == max(supports) for value in supports) != 1
        or evidence.preferred_value_id != decision.preference_value_id
    ):
        return finish(MemoryDecisionAction.ASK, CADENCE_MEMORY_AUTO_THRESHOLD_NOT_MET)
    if not _active_partition_is_valid(mapping, active_node_ids):
        return finish(MemoryDecisionAction.ASK, "MEMORY_CANDIDATE_MAPPING_INVALID")
    option = mapping.option_for_preference(decision.preference_value_id or "")
    if option is None or option.option_id != decision.option_id:
        return finish(MemoryDecisionAction.ASK, "MEMORY_CANDIDATE_MAPPING_INVALID")
    if len(option.mapped_node_ids) >= len(active_node_ids):
        return finish(MemoryDecisionAction.ASK, CADENCE_MEMORY_OPTION_DOES_NOT_NARROW)

    option_ids = {item.option_id for item in mapping.options}
    assessments_valid = (
        isinstance(option_intent_assessments, Mapping)
        and set(option_intent_assessments) == option_ids
        and all(
            isinstance(key, str)
            and isinstance(value, str)
            and value in intent_labels
            for key, value in option_intent_assessments.items()
        )
    )
    if not assessments_valid:
        return finish(
            MemoryDecisionAction.ASK,
            CADENCE_MEMORY_CURRENT_INTENT_ASSESSMENT_INVALID,
        )
    selected_assessment = option_intent_assessments[option.option_id]
    another_supported = any(
        option_id != option.option_id
        and assessment == CADENCE_INTENT_SUPPORTED_STATUS
        for option_id, assessment in option_intent_assessments.items()
    )
    if selected_assessment in CADENCE_INTENT_BLOCKING_STATUSES or (
        selected_assessment == CADENCE_INTENT_COMPATIBLE_STATUS
        and CADENCE_COMPATIBLE_BLOCKED_BY_OTHER_SUPPORTED
        and another_supported
    ):
        return finish(
            MemoryDecisionAction.ASK,
            CADENCE_MEMORY_CURRENT_INTENT_CONFLICT,
        )
    return finish(MemoryDecisionAction.AUTO_SELECT, decision.reason_code)
