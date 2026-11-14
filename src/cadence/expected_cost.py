"""精确期望提问成本 selector。

本模块只消费已通过归因资格、scope、partition 和 renderability 检查的完整 probe
分区；它不生成问题、不扩展候选，也不把非法/未映射回答加入概率模型。DP 状态是
当前活动 semantic class 的子集，回答按冻结分区与目标 class 保留交集。浮点数会让
期望成本并列结果依赖舍入误差，因此所有概率和成本使用 ``Fraction`` 精确比较。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache

from cadence.constants import CADENCE_FACTOR_ORDER

CADENCE_DP_ACTIVE_NODES_INVALID = "CADENCE_DP_ACTIVE_NODES_INVALID"
CADENCE_DP_FACTOR_TYPE_UNREGISTERED = "CADENCE_DP_FACTOR_TYPE_UNREGISTERED"
CADENCE_DP_PARTITION_INVALID = "CADENCE_DP_PARTITION_INVALID"
CADENCE_DP_TIEBREAK_AMBIGUOUS = "CADENCE_DP_TIEBREAK_AMBIGUOUS"


class ExpectedCostSelectionError(ValueError):
    """selector 的稳定输入/方法失败；reason_code 可供实验入口处理。"""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(reason_code if not detail else f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class ProbePartition:
    """一个已渲染 probe 在初始活动视图上的完整语义选项分区。

    ``query_scope_order`` 是 AST/algebra 左到右遍历产生的整数路径；不能用 scope
    名称、哈希或候选输入顺序代替。每个可选项只包含其映射到的 semantic class ID；
    EXIT/无效反馈不属于这个 ASK-only 成本模型。
    """

    probe_id: str
    factor_type: str
    query_scope_order: tuple[int, ...]
    slot_order: int
    groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ExpectedCostSelection:
    """一次纯 DP 决策；不可识别时显式返回 UNRESOLVED。"""

    status: str
    probe_id: str | None
    expected_cost: Fraction | None
    max_child_size: int | None
    states_evaluated: int


def _normalize_active_nodes(active_node_ids: Sequence[str]) -> tuple[str, ...]:
    """按 UTF-8 字节稳定排序 class ID，得到唯一 bit 位分配。"""

    if (
        isinstance(active_node_ids, (str, bytes))
        or not isinstance(active_node_ids, Sequence)
        or not active_node_ids
        or any(not isinstance(item, str) or not item for item in active_node_ids)
        or len(set(active_node_ids)) != len(active_node_ids)
    ):
        raise ExpectedCostSelectionError(CADENCE_DP_ACTIVE_NODES_INVALID)
    return tuple(sorted(active_node_ids, key=lambda item: item.encode("utf-8")))


def _normalize_candidates(
    candidates: Sequence[ProbePartition],
    active: tuple[str, ...],
    factor_order: tuple[str, ...],
) -> tuple[tuple[ProbePartition, tuple[int, ...]], ...]:
    """校验每个 probe 都是完整、不重叠且有区分力的 active-view 分区。"""

    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise ExpectedCostSelectionError(CADENCE_DP_PARTITION_INVALID, "candidate sequence")
    node_bits = {node_id: 1 << index for index, node_id in enumerate(active)}
    factor_ranks = {factor: index for index, factor in enumerate(factor_order)}
    seen_probe_ids: set[str] = set()
    normalized: list[tuple[ProbePartition, tuple[int, ...]]] = []
    for candidate in candidates:
        if not isinstance(candidate, ProbePartition):
            raise ExpectedCostSelectionError(CADENCE_DP_PARTITION_INVALID, "candidate type")
        if not isinstance(candidate.probe_id, str) or not candidate.probe_id:
            raise ExpectedCostSelectionError(CADENCE_DP_PARTITION_INVALID, "probe_id")
        if candidate.probe_id in seen_probe_ids:
            raise ExpectedCostSelectionError(CADENCE_DP_PARTITION_INVALID, "duplicate probe_id")
        seen_probe_ids.add(candidate.probe_id)
        if (
            not isinstance(candidate.factor_type, str)
            or candidate.factor_type not in factor_ranks
        ):
            raise ExpectedCostSelectionError(
                CADENCE_DP_FACTOR_TYPE_UNREGISTERED, str(candidate.factor_type)
            )
        if (
            not isinstance(candidate.query_scope_order, tuple)
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in candidate.query_scope_order
            )
            or isinstance(candidate.slot_order, bool)
            or not isinstance(candidate.slot_order, int)
            or candidate.slot_order < 0
        ):
            raise ExpectedCostSelectionError(CADENCE_DP_PARTITION_INVALID, "coordinate order")
        if (
            not isinstance(candidate.groups, tuple)
            or len(candidate.groups) < 2
            or any(
                not isinstance(group, tuple)
                or not group
                or any(not isinstance(node_id, str) or not node_id for node_id in group)
                for group in candidate.groups
            )
        ):
            raise ExpectedCostSelectionError(CADENCE_DP_PARTITION_INVALID, "empty partition cell")
        seen_nodes: set[str] = set()
        group_masks: list[int] = []
        for group in candidate.groups:
            if len(set(group)) != len(group) or any(node not in node_bits for node in group):
                raise ExpectedCostSelectionError(
                    CADENCE_DP_PARTITION_INVALID, "unknown/duplicate class"
                )
            if seen_nodes.intersection(group):
                raise ExpectedCostSelectionError(CADENCE_DP_PARTITION_INVALID, "overlapping cells")
            seen_nodes.update(group)
            group_masks.append(sum(node_bits[node] for node in group))
        if seen_nodes != set(active):
            raise ExpectedCostSelectionError(CADENCE_DP_PARTITION_INVALID, "incomplete coverage")
        normalized.append((candidate, tuple(group_masks)))
    return tuple(normalized)


def select_expected_cost_probe(
    *,
    active_node_ids: Sequence[str],
    candidates: Sequence[ProbePartition],
    excluded_initial_probe_ids: Sequence[str] = (),
) -> ExpectedCostSelection:
    """选择达到 singleton 的精确期望提问数最小的首个 probe。

    状态只由当前 class 子集决定；每个 probe 在子状态上的结果是其冻结选项组与该
    子集的交集。若所有合法决策树都无法把每个分支缩至 singleton，返回
    ``UNRESOLVED``，不退化为贪心、MINIMAX 或候选补全。复杂度最坏为 ``O(2^n * m * n)``，
    其中 n 是当前 semantic class 数，m 是初始合法 probe 数。
    """

    active = _normalize_active_nodes(active_node_ids)
    if (
        isinstance(excluded_initial_probe_ids, (str, bytes))
        or not isinstance(excluded_initial_probe_ids, Sequence)
        or any(not isinstance(item, str) or not item for item in excluded_initial_probe_ids)
        or len(set(excluded_initial_probe_ids)) != len(excluded_initial_probe_ids)
    ):
        raise ExpectedCostSelectionError(CADENCE_DP_PARTITION_INVALID, "excluded probe IDs")
    probes = _normalize_candidates(candidates, active, CADENCE_FACTOR_ORDER)
    probe_ids = {probe.probe_id for probe, _groups in probes}
    excluded = set(excluded_initial_probe_ids)
    if not excluded.issubset(probe_ids):
        raise ExpectedCostSelectionError(CADENCE_DP_PARTITION_INVALID, "unknown excluded probe ID")
    full_mask = (1 << len(active)) - 1
    states: set[int] = set()
    factor_ranks = {factor: index for index, factor in enumerate(CADENCE_FACTOR_ORDER)}

    def children_for(state: int, group_masks: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(child for mask in group_masks if (child := state & mask))

    @cache
    def minimum_cost(state: int) -> Fraction | None:
        states.add(state)
        if state.bit_count() == 1:
            return Fraction(0, 1)
        best: Fraction | None = None
        for _probe, group_masks in probes:
            children = children_for(state, group_masks)
            if len(children) < 2:
                continue
            child_costs = tuple(minimum_cost(child) for child in children)
            if any(cost is None for cost in child_costs):
                continue
            expected = Fraction(1, 1) + sum(
                (
                    Fraction(child.bit_count(), state.bit_count()) * cost
                    for child, cost in zip(children, child_costs, strict=True)
                    if cost is not None
                ),
                Fraction(0, 1),
            )
            if best is None or expected < best:
                best = expected
        return best

    optimum = minimum_cost(full_mask)
    if len(active) == 1:
        return ExpectedCostSelection("IDENTIFIED", None, Fraction(0, 1), None, len(states))
    if optimum is None:
        return ExpectedCostSelection("UNRESOLVED", None, None, None, len(states))

    root_options: list[
        tuple[Fraction, int, int, tuple[int, ...], int, ProbePartition, tuple[int, ...]]
    ] = []
    for probe, group_masks in probes:
        # 问句生成失败只排除当前活动视图中的首个选择；递归子状态仍按完整
        # 可行 probe universe 估计成本，因为回答后会重建分区并重新运行 LLM 检查。
        if probe.probe_id in excluded:
            continue
        children = children_for(full_mask, group_masks)
        if len(children) < 2:
            continue
        child_costs = tuple(minimum_cost(child) for child in children)
        if any(cost is None for cost in child_costs):
            continue
        cost = Fraction(1, 1) + sum(
            (
                Fraction(child.bit_count(), len(active)) * child_cost
                for child, child_cost in zip(children, child_costs, strict=True)
                if child_cost is not None
            ),
            Fraction(0, 1),
        )
        max_child_size = max(child.bit_count() for child in children)
        root_options.append(
            (
                cost,
                max_child_size,
                factor_ranks[probe.factor_type],
                probe.query_scope_order,
                probe.slot_order,
                probe,
                group_masks,
            )
        )

    if not root_options:
        return ExpectedCostSelection("UNRESOLVED", None, None, None, len(states))
    best_cost = min(item[0] for item in root_options)
    root_options = [item for item in root_options if item[0] == best_cost]
    # 完整 root identification key 使用固定类型、scope 与 semantic slot 顺序，
    # 保证并列 probe 的选择稳定且可复现。
    root_options.sort(key=lambda item: item[:5])
    best = root_options[0]
    best_tie_key = best[:5]
    tied = [item for item in root_options if item[:5] == best_tie_key]
    if len(tied) > 1:
        raise ExpectedCostSelectionError(
            CADENCE_DP_TIEBREAK_AMBIGUOUS,
            "根类型、scope 与 semantic slot 顺序不足以区分不同 probe",
        )
    return ExpectedCostSelection(
        status="SELECTED",
        probe_id=best[5].probe_id,
        expected_cost=best[0],
        max_child_size=best[1],
        states_evaluated=len(states),
    )

