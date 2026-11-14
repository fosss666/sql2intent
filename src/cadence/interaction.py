"""CADENCE 交互循环的确定性控制器；LLM、用户界面与记忆持久化由调用方注入。

控制器只在已分组候选的根分区上循环，不生成 SQL 或新增候选。问句不可行时，
仅排除当前活动视图中的该分区；显式回答或记忆自动选择后重新计算活动视图。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from cadence.constants import CADENCE_INTENT_ASSESSMENT_LABELS
from cadence.expected_cost import ExpectedCostSelectionError
from cadence.grouping import (
    CadenceGroupingError,
    FactorizedCandidate,
    RootPartition,
    SemanticClass,
)
from cadence.memory import (
    CandidateOptionMapping,
    MemoryConfig,
    MemoryDecision,
    MemoryDecisionAction,
    MemoryEngine,
    MemoryScope,
    PreferenceMemoryError,
)
from cadence.memory_policy import (
    CadenceMemoryPolicyError,
    decide_cadence_memory,
)
from cadence.pipeline import CadencePipelineError, select_next_partition

CADENCE_INTERACTION_INPUT_INVALID = "CADENCE_INTERACTION_INPUT_INVALID"
CADENCE_INTERACTION_ATTRIBUTION_FAILED = "CADENCE_INTERACTION_ATTRIBUTION_FAILED"
CADENCE_INTERACTION_ATTRIBUTION_INVALID = "CADENCE_INTERACTION_ATTRIBUTION_INVALID"
CADENCE_INTERACTION_OPTION_ASSESSMENT_FAILED = "CADENCE_INTERACTION_OPTION_ASSESSMENT_FAILED"
CADENCE_INTERACTION_OPTIONS_INVALID = "CADENCE_INTERACTION_OPTIONS_INVALID"
CADENCE_INTERACTION_QUESTION_FAILED = "CADENCE_INTERACTION_QUESTION_FAILED"
CADENCE_INTERACTION_QUESTION_INVALID = "CADENCE_INTERACTION_QUESTION_INVALID"
CADENCE_INTERACTION_BUDGET_EXHAUSTED = "CADENCE_INTERACTION_BUDGET_EXHAUSTED"
CADENCE_INTERACTION_NO_FEASIBLE_PARTITION = "CADENCE_INTERACTION_NO_FEASIBLE_PARTITION"
CADENCE_INTERACTION_MEMORY_FAILED = "CADENCE_INTERACTION_MEMORY_FAILED"
CADENCE_INTERACTION_ANSWER_FAILED = "CADENCE_INTERACTION_ANSWER_FAILED"
CADENCE_INTERACTION_USER_ABSTAIN = "CADENCE_INTERACTION_USER_ABSTAIN"
CADENCE_INTERACTION_ANSWER_INVALID = "CADENCE_INTERACTION_ANSWER_INVALID"
CADENCE_INTERACTION_FEEDBACK_SINK_REQUIRED = "CADENCE_INTERACTION_FEEDBACK_SINK_REQUIRED"
CADENCE_INTERACTION_FEEDBACK_WRITE_FAILED = "CADENCE_INTERACTION_FEEDBACK_WRITE_FAILED"


@dataclass(frozen=True)
class CadenceAttributionInput:
    """供语义归因器使用的当前根、分区和活动语义类。"""

    request_text: str
    partition: RootPartition
    active_class_ids: tuple[str, ...]
    semantic_classes: tuple[SemanticClass, ...]


@dataclass(frozen=True)
class CadenceAttribution:
    """LLM 对当前根的自然语言证据、选项映射和意图一致性判断。

    ``preference_value_id`` 必须是解释含义的稳定规范 ID，而不是展示文案；Memory
    用完整 ID 集合比较历史选项，若同义表达未映射到同一 ID，会错误地拒绝可复用记录。
    """

    ambiguity_text: str
    context_key: str | None
    context_text: str | None
    option_mapping: CandidateOptionMapping
    option_interpretations: Mapping[str, str]
    option_intent_assessments: Mapping[str, str]


@dataclass(frozen=True)
class CadenceOptionAssessmentInput:
    """供 LLM 生成并检查自然语言选项；此阶段不生成最终问句。"""

    request_text: str
    partition: RootPartition
    attribution: CadenceAttribution


@dataclass(frozen=True)
class CadenceOptionAssessment:
    """逐组选项及语义相关、意图可区分、用户可回答三项检查结果。"""

    option_texts: Mapping[str, str]
    semantic_relevance: bool
    semantic_distinguishability: bool
    user_answerability: bool


@dataclass(frozen=True)
class CadenceQuestionInput:
    """仅在选项通过三项检查后传给最终问句生成器的数据。"""

    request_text: str
    partition: RootPartition
    attribution: CadenceAttribution
    option_texts: Mapping[str, str]
    questions_remaining: int


@dataclass(frozen=True)
class CadenceQuestionDraft:
    """最终问句；其选项文案必须与已检查的选项完全相同。"""

    question_text: str
    option_texts: Mapping[str, str]


@dataclass(frozen=True)
class CadenceMemoryBinding:
    """本轮 Memory 的显式运行参数，避免依赖隐式全局配置。"""

    engine: MemoryEngine
    scope: MemoryScope
    config: MemoryConfig
    m_min: float
    as_of: datetime


@dataclass(frozen=True)
class CadenceFeedbackInput:
    """只在用户对已展示问句给出明确选项后交给持久化适配器。"""

    request_text: str
    partition: RootPartition
    attribution: CadenceAttribution
    question: CadenceQuestionDraft
    option_assessment: CadenceOptionAssessment
    chosen_option_id: str
    candidate_ids: tuple[str, ...]
    round_index: int
    memory_decision: MemoryDecision | None
    as_of: datetime | None


@dataclass(frozen=True)
class CadenceInteractionResult:
    """交互的确定性终态和活动视图；不包含也不重新生成 SQL。"""

    status: Literal["IDENTIFIED", "ABSTAIN"]
    reason_code: str | None
    active_class_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    questions_asked: int
    memory_decisions: tuple[MemoryDecision, ...]
    feedback_failures: tuple[str, ...]


AttributionCallback = Callable[[CadenceAttributionInput], CadenceAttribution]
OptionAssessmentCallback = Callable[
    [CadenceOptionAssessmentInput], CadenceOptionAssessment
]
QuestionCallback = Callable[[CadenceQuestionInput], CadenceQuestionDraft]
AnswerCallback = Callable[[CadenceQuestionDraft], str | None]
FeedbackCallback = Callable[[CadenceFeedbackInput], None]


def _result(
    *,
    status: Literal["IDENTIFIED", "ABSTAIN"],
    reason_code: str | None,
    active_class_ids: Sequence[str],
    candidate_ids: Sequence[str] = (),
    questions_asked: int,
    memory_decisions: Sequence[MemoryDecision],
    feedback_failures: Sequence[str],
) -> CadenceInteractionResult:
    """固定输出顺序，使交互结果可被上层稳定记录。"""

    return CadenceInteractionResult(
        status=status,
        reason_code=reason_code,
        active_class_ids=tuple(active_class_ids),
        candidate_ids=tuple(candidate_ids),
        questions_asked=questions_asked,
        memory_decisions=tuple(memory_decisions),
        feedback_failures=tuple(feedback_failures),
    )


def _attribution_is_valid(
    attribution: CadenceAttribution,
    partition: RootPartition,
    active_class_ids: Sequence[str],
    *,
    require_intent_assessments: bool,
) -> bool:
    """确保归因选项完整互斥，阻止问句或 Memory 映射扩展当前候选池。"""

    if (
        not isinstance(attribution.ambiguity_text, str)
        or not attribution.ambiguity_text.strip()
        or any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in (attribution.context_key, attribution.context_text)
        )
        or not isinstance(attribution.option_mapping, CandidateOptionMapping)
    ):
        return False
    expected_groups = {frozenset(group) for group in partition.groups}
    actual_groups = {
        frozenset(option.mapped_node_ids)
        for option in attribution.option_mapping.options
    }
    if (
        len(expected_groups) != len(partition.groups)
        or len(actual_groups) != len(attribution.option_mapping.options)
        or actual_groups != expected_groups
        or set().union(*actual_groups) != set(active_class_ids)
    ):
        return False

    option_ids = {item.option_id for item in attribution.option_mapping.options}
    interpretations = attribution.option_interpretations
    assessments = attribution.option_intent_assessments
    if (
        not isinstance(interpretations, Mapping)
        or set(interpretations) != option_ids
        or any(
            not isinstance(value, str) or not value.strip()
            for value in interpretations.values()
        )
        or not isinstance(assessments, Mapping)
    ):
        return False
    if not assessments and not require_intent_assessments:
        return True
    allowed_assessments = set(CADENCE_INTENT_ASSESSMENT_LABELS)
    return set(assessments) == option_ids and all(
        isinstance(value, str) and value in allowed_assessments
        for value in assessments.values()
    )


def _option_assessment_is_valid(
    assessment: CadenceOptionAssessment,
    attribution: CadenceAttribution,
) -> bool:
    """在问句生成前验证选项集合和三项可行性检查的结构。"""

    return (
        isinstance(assessment.option_texts, Mapping)
        and set(assessment.option_texts) == set(attribution.option_interpretations)
        and all(
            isinstance(value, str) and value.strip()
            for value in assessment.option_texts.values()
        )
        and all(
            isinstance(value, bool)
            for value in (
                assessment.semantic_relevance,
                assessment.semantic_distinguishability,
                assessment.user_answerability,
            )
        )
    )


def _question_is_valid(
    question: CadenceQuestionDraft,
    assessment: CadenceOptionAssessment,
) -> bool:
    """防止最终问句在通过评估后改写、补充或删除已冻结选项。"""

    return (
        isinstance(question.question_text, str)
        and bool(question.question_text.strip())
        and isinstance(question.option_texts, Mapping)
        and dict(question.option_texts) == dict(assessment.option_texts)
    )


def run_cadence_interaction(
    *,
    candidates: Sequence[FactorizedCandidate],
    request_text: str,
    attribute: AttributionCallback,
    assess_options: OptionAssessmentCallback,
    generate_question: QuestionCallback,
    answer: AnswerCallback,
    max_questions: int,
    memory: CadenceMemoryBinding | None = None,
    record_feedback: FeedbackCallback | None = None,
) -> CadenceInteractionResult:
    """执行交互闭环：DP 选根、LLM 检查、Memory 与显式回答后收缩活动视图。

    LLM 与用户界面由显式回调承担，因此相同输入/回调结果产生相同控制流。问句
    不可行时只排除当前视图中的根分区并重跑 DP；预算耗尽或没有可行分区时弃答。
    """

    empty: tuple[str, ...] = ()
    empty_memory_decisions: tuple[MemoryDecision, ...] = ()
    if (
        isinstance(max_questions, bool)
        or not isinstance(max_questions, int)
        or max_questions < 0
        or not isinstance(request_text, str)
        or not request_text.strip()
        or not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or not candidates
        or not callable(attribute)
        or not callable(assess_options)
        or not callable(generate_question)
        or not callable(answer)
    ):
        return _result(
            status="ABSTAIN",
            reason_code=CADENCE_INTERACTION_INPUT_INVALID,
            active_class_ids=empty,
            questions_asked=0,
            memory_decisions=empty_memory_decisions,
            feedback_failures=empty,
        )
    if memory is not None and not callable(record_feedback):
        return _result(
            status="ABSTAIN",
            reason_code=CADENCE_INTERACTION_FEEDBACK_SINK_REQUIRED,
            active_class_ids=empty,
            questions_asked=0,
            memory_decisions=empty_memory_decisions,
            feedback_failures=empty,
        )

    questions_asked = 0
    active_class_ids: tuple[str, ...] | None = None
    excluded_partition_ids: tuple[str, ...] = ()
    memory_decisions: list[MemoryDecision] = []
    feedback_failures: list[str] = []

    while True:
        try:
            selection = select_next_partition(
                candidates,
                active_class_ids=active_class_ids,
                excluded_partition_ids=excluded_partition_ids,
            )
        except (CadencePipelineError, CadenceGroupingError, ExpectedCostSelectionError) as exc:
            return _result(
                status="ABSTAIN",
                reason_code=exc.reason_code,
                active_class_ids=active_class_ids or empty,
                questions_asked=questions_asked,
                memory_decisions=memory_decisions,
                feedback_failures=feedback_failures,
            )

        active_class_ids = selection.active_class_ids
        if len(active_class_ids) == 1:
            identified = next(
                item
                for item in selection.semantic_classes
                if item.class_id == active_class_ids[0]
            )
            return _result(
                status="IDENTIFIED",
                reason_code=None,
                active_class_ids=active_class_ids,
                candidate_ids=identified.candidate_ids,
                questions_asked=questions_asked,
                memory_decisions=memory_decisions,
                feedback_failures=feedback_failures,
            )
        partition = selection.selected_partition
        if partition is None:
            return _result(
                status="ABSTAIN",
                reason_code=CADENCE_INTERACTION_NO_FEASIBLE_PARTITION,
                active_class_ids=active_class_ids,
                questions_asked=questions_asked,
                memory_decisions=memory_decisions,
                feedback_failures=feedback_failures,
            )

        try:
            attribution = attribute(
                CadenceAttributionInput(
                    request_text=request_text,
                    partition=partition,
                    active_class_ids=active_class_ids,
                    semantic_classes=selection.semantic_classes,
                )
            )
        except Exception:
            return _result(
                status="ABSTAIN",
                reason_code=CADENCE_INTERACTION_ATTRIBUTION_FAILED,
                active_class_ids=active_class_ids,
                questions_asked=questions_asked,
                memory_decisions=memory_decisions,
                feedback_failures=feedback_failures,
            )
        if not isinstance(attribution, CadenceAttribution) or not _attribution_is_valid(
            attribution,
            partition,
            active_class_ids,
            require_intent_assessments=memory is not None,
        ):
            return _result(
                status="ABSTAIN",
                reason_code=CADENCE_INTERACTION_ATTRIBUTION_INVALID,
                active_class_ids=active_class_ids,
                questions_asked=questions_asked,
                memory_decisions=memory_decisions,
                feedback_failures=feedback_failures,
            )

        memory_decision: MemoryDecision | None = None
        if memory is not None:
            try:
                memory_decision = decide_cadence_memory(
                    memory.engine,
                    scope=memory.scope,
                    query_ambiguity_key=partition.factor_type,
                    query_ambiguity_text=attribution.ambiguity_text,
                    query_context_key=attribution.context_key,
                    query_context_text=attribution.context_text,
                    mapping=attribution.option_mapping,
                    config=memory.config,
                    m_min=memory.m_min,
                    active_node_ids=active_class_ids,
                    option_intent_assessments=attribution.option_intent_assessments,
                    as_of=memory.as_of,
                )
            except (CadenceMemoryPolicyError, PreferenceMemoryError) as exc:
                reason_code = getattr(exc, "reason_code", CADENCE_INTERACTION_MEMORY_FAILED)
                return _result(
                    status="ABSTAIN",
                    reason_code=reason_code,
                    active_class_ids=active_class_ids,
                    questions_asked=questions_asked,
                    memory_decisions=memory_decisions,
                    feedback_failures=feedback_failures,
                )
            memory_decisions.append(memory_decision)
            if memory_decision.action is MemoryDecisionAction.AUTO_SELECT:
                option = next(
                    (
                        item
                        for item in attribution.option_mapping.options
                        if item.option_id == memory_decision.option_id
                    ),
                    None,
                )
                if option is None:
                    return _result(
                        status="ABSTAIN",
                        reason_code=CADENCE_INTERACTION_MEMORY_FAILED,
                        active_class_ids=active_class_ids,
                        questions_asked=questions_asked,
                        memory_decisions=memory_decisions,
                        feedback_failures=feedback_failures,
                    )
                active_class_ids = option.mapped_node_ids
                # Memory 自动选择不是显式反馈；不写入新的偏好事件。
                excluded_partition_ids = ()
                continue

        if questions_asked >= max_questions:
            return _result(
                status="ABSTAIN",
                reason_code=CADENCE_INTERACTION_BUDGET_EXHAUSTED,
                active_class_ids=active_class_ids,
                questions_asked=questions_asked,
                memory_decisions=memory_decisions,
                feedback_failures=feedback_failures,
            )

        try:
            option_assessment = assess_options(
                CadenceOptionAssessmentInput(
                    request_text=request_text,
                    partition=partition,
                    attribution=attribution,
                )
            )
        except Exception:
            return _result(
                status="ABSTAIN",
                reason_code=CADENCE_INTERACTION_OPTION_ASSESSMENT_FAILED,
                active_class_ids=active_class_ids,
                questions_asked=questions_asked,
                memory_decisions=memory_decisions,
                feedback_failures=feedback_failures,
            )
        if not isinstance(
            option_assessment, CadenceOptionAssessment
        ) or not _option_assessment_is_valid(option_assessment, attribution):
            return _result(
                status="ABSTAIN",
                reason_code=CADENCE_INTERACTION_OPTIONS_INVALID,
                active_class_ids=active_class_ids,
                questions_asked=questions_asked,
                memory_decisions=memory_decisions,
                feedback_failures=feedback_failures,
            )
        if not (
            option_assessment.semantic_relevance
            and option_assessment.semantic_distinguishability
            and option_assessment.user_answerability
        ):
            excluded_partition_ids = (*excluded_partition_ids, partition.partition_id)
            continue

        try:
            question = generate_question(
                CadenceQuestionInput(
                    request_text=request_text,
                    partition=partition,
                    attribution=attribution,
                    option_texts=option_assessment.option_texts,
                    questions_remaining=max_questions - questions_asked,
                )
            )
        except Exception:
            return _result(
                status="ABSTAIN",
                reason_code=CADENCE_INTERACTION_QUESTION_FAILED,
                active_class_ids=active_class_ids,
                questions_asked=questions_asked,
                memory_decisions=memory_decisions,
                feedback_failures=feedback_failures,
            )
        if not isinstance(question, CadenceQuestionDraft) or not _question_is_valid(
            question, option_assessment
        ):
            return _result(
                status="ABSTAIN",
                reason_code=CADENCE_INTERACTION_QUESTION_INVALID,
                active_class_ids=active_class_ids,
                questions_asked=questions_asked,
                memory_decisions=memory_decisions,
                feedback_failures=feedback_failures,
            )

        try:
            chosen_option_id = answer(question)
        except Exception:
            return _result(
                status="ABSTAIN",
                reason_code=CADENCE_INTERACTION_ANSWER_FAILED,
                active_class_ids=active_class_ids,
                questions_asked=questions_asked,
                memory_decisions=memory_decisions,
                feedback_failures=feedback_failures,
            )
        if chosen_option_id is None:
            return _result(
                status="ABSTAIN",
                reason_code=CADENCE_INTERACTION_USER_ABSTAIN,
                active_class_ids=active_class_ids,
                questions_asked=questions_asked,
                memory_decisions=memory_decisions,
                feedback_failures=feedback_failures,
            )
        option = next(
            (
                item
                for item in attribution.option_mapping.options
                if item.option_id == chosen_option_id
            ),
            None,
        )
        if option is None:
            return _result(
                status="ABSTAIN",
                reason_code=CADENCE_INTERACTION_ANSWER_INVALID,
                active_class_ids=active_class_ids,
                questions_asked=questions_asked,
                memory_decisions=memory_decisions,
                feedback_failures=feedback_failures,
            )

        questions_asked += 1
        if memory is not None and record_feedback is not None:
            try:
                candidate_ids = tuple(
                    candidate_id
                    for semantic_class in selection.semantic_classes
                    if semantic_class.class_id in active_class_ids
                    for candidate_id in semantic_class.candidate_ids
                )
                record_feedback(
                    CadenceFeedbackInput(
                        request_text=request_text,
                        partition=partition,
                        attribution=attribution,
                        question=question,
                        option_assessment=option_assessment,
                        chosen_option_id=chosen_option_id,
                        candidate_ids=candidate_ids,
                        round_index=questions_asked - 1,
                        memory_decision=memory_decision,
                        as_of=memory.as_of,
                    )
                )
            except Exception:
                # 用户已给出的明确答案仍然收缩活动视图；记录失败单独暴露给运行层。
                feedback_failures.append(CADENCE_INTERACTION_FEEDBACK_WRITE_FAILED)

        active_class_ids = option.mapped_node_ids
        # 根排除只对当前选择有效；回答改变了视图，后续分区须从头构造。
        excluded_partition_ids = ()
