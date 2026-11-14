"""候选分组与 DP 的组合入口。

上游提供规范化完整 factorization。初次选择先对所有非平凡根分区运行 DP；若所选根
无法生成可行问句，只从当前活动视图的首选项中排除该根并重选。回答后应重建活动分区，
再调用本模块；这避免把一次局部问句失败永久传播到后续子视图。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cadence.expected_cost import (
    ExpectedCostSelection,
    ProbePartition,
    select_expected_cost_probe,
)
from cadence.grouping import (
    FactorizedCandidate,
    RootPartition,
    SemanticClass,
    build_root_partitions,
    group_equivalent_candidates,
)

CADENCE_EXCLUDED_PROBE_UNKNOWN = "CADENCE_EXCLUDED_PROBE_UNKNOWN"
CADENCE_EXCLUDED_PROBE_INVALID = "CADENCE_EXCLUDED_PROBE_INVALID"


class CadencePipelineError(ValueError):
    """CADENCE 组合入口的稳定输入失败。"""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class CadencePartitionSelection:
    """保留本轮分组、可选根分区与 DP 结果，便于实验层记录和审计。"""

    semantic_classes: tuple[SemanticClass, ...]
    active_class_ids: tuple[str, ...]
    candidate_partitions: tuple[RootPartition, ...]
    selection: ExpectedCostSelection
    selected_partition: RootPartition | None


def select_next_partition(
    candidates: Sequence[FactorizedCandidate],
    *,
    active_class_ids: Sequence[str] | None = None,
    excluded_partition_ids: Sequence[str] = (),
) -> CadencePartitionSelection:
    """在完整根分区 universe 上执行分组与精确 DP，再返回当前最优分区。

    初次调用覆盖全部非平凡根分区。若后续归因/问句生成判定所选根不可行，
    调用方把当前分区 ID 加入 ``excluded_partition_ids`` 后再次调用。该排除只作用于
    当前视图的首个选择；期望成本的后续子状态仍能考虑完整 universe。回答后必须基于
    新活动视图重新生成 partition ID，旧排除不应跨轮沿用。
    """

    if (
        isinstance(excluded_partition_ids, (str, bytes))
        or not isinstance(excluded_partition_ids, Sequence)
        or any(not isinstance(item, str) or not item for item in excluded_partition_ids)
        or len(set(excluded_partition_ids)) != len(excluded_partition_ids)
    ):
        raise CadencePipelineError(CADENCE_EXCLUDED_PROBE_INVALID)

    semantic_classes = group_equivalent_candidates(candidates)
    if active_class_ids is None:
        active = tuple(item.class_id for item in semantic_classes)
    else:
        if (
            isinstance(active_class_ids, (str, bytes))
            or not isinstance(active_class_ids, Sequence)
            or any(not isinstance(item, str) or not item for item in active_class_ids)
            or len(set(active_class_ids)) != len(active_class_ids)
        ):
            raise CadencePipelineError(CADENCE_EXCLUDED_PROBE_INVALID)
        active = tuple(active_class_ids)

    generated = build_root_partitions(semantic_classes, active_class_ids=active)
    generated_by_id = {item.partition_id: item for item in generated}
    if len(generated_by_id) != len(generated):
        raise CadencePipelineError(CADENCE_EXCLUDED_PROBE_INVALID)
    unknown = set(excluded_partition_ids).difference(generated_by_id)
    if unknown:
        raise CadencePipelineError(CADENCE_EXCLUDED_PROBE_UNKNOWN)

    excluded_id_set = set(excluded_partition_ids)
    probe_partitions: tuple[ProbePartition, ...] = tuple(
        partition.as_probe_partition() for partition in generated
    )
    selection = select_expected_cost_probe(
        active_node_ids=active,
        candidates=probe_partitions,
        excluded_initial_probe_ids=tuple(
            sorted(excluded_id_set, key=lambda item: item.encode("utf-8"))
        ),
    )
    selected = generated_by_id.get(selection.probe_id) if selection.probe_id else None
    return CadencePartitionSelection(
        semantic_classes=semantic_classes,
        active_class_ids=tuple(sorted(active, key=lambda item: item.encode("utf-8"))),
        candidate_partitions=generated,
        selection=selection,
        selected_partition=selected,
    )
