"""构造候选因子等价类和语义根分区。

完整 factorized representation（roots、derived、residual）决定候选等价类；仅在
此后，semantic root coordinate 的取值才生成澄清分区。Residual 差异不能丢弃，缺失
坐标显式作为独立值。本模块不推断 SQL scope/slot，也不生成 SQL。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from cadence.canonical_serialize import CanonicalSerializeError, canonical_bytes
from cadence.constants import CADENCE_FACTOR_ORDER
from cadence.expected_cost import ProbePartition

CADENCE_FACTORIZATION_INVALID = "CADENCE_FACTORIZATION_INVALID"
CADENCE_FACTOR_TYPE_UNREGISTERED = "CADENCE_FACTOR_TYPE_UNREGISTERED"


class CadenceGroupingError(ValueError):
    """Factorization 或 root partition 的稳定失败。"""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(reason_code if not detail else f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class RootCoordinate:
    """一个 query scope/semantic slot 中实际存在的根坐标及其规范值。

    根坐标缺失由候选的 ``roots`` 中没有该 key 表示，并作为独立的 ABSENT 值；抽取
    失败必须由上游拒绝，不能伪装成 ABSENT 后参与分组。
    """

    factor_type: str
    scope_id: str
    scope_order: tuple[int, ...]
    slot_id: str
    slot_order: int
    canonical_value: Any


@dataclass(frozen=True)
class FactorizedCandidate:
    """单个候选的完整分解；candidate ID 不参与等价类 identity。"""

    candidate_id: str
    roots: tuple[RootCoordinate, ...]
    derived: Any
    residual: Any


@dataclass(frozen=True)
class SemanticClass:
    """共享完整 factorized representation 的有序候选集合。"""

    class_id: str
    candidate_ids: tuple[str, ...]
    roots: tuple[RootCoordinate, ...]
    derived: Any
    residual: Any


@dataclass(frozen=True)
class RootPartition:
    """由一个 root coordinate 导出的完整活动 class 分区。"""

    partition_id: str
    factor_type: str
    scope_id: str
    slot_id: str
    query_scope_order: tuple[int, ...]
    slot_order: int
    groups: tuple[tuple[str, ...], ...]

    def as_probe_partition(self) -> ProbePartition:
        """将分区传给 DP；调用方仍须先完成归因与 renderability 准入。"""

        return ProbePartition(
            probe_id=self.partition_id,
            factor_type=self.factor_type,
            query_scope_order=self.query_scope_order,
            slot_order=self.slot_order,
            groups=self.groups,
        )


def _json_value(value: Any, *, path: str) -> Any:
    """将 tuple 等 Python 容器归一化为 JSON array/object，再由 canonical bytes 验收。"""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        # identity 使用 typed string 表示，避免不同 JSON 实现对数值的编码差异。
        raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, f"numeric token at {path}")
    if isinstance(value, float):
        raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, f"float token at {path}")
    if isinstance(value, (list, tuple)):
        return [_json_value(item, path=f"{path}[]") for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, f"non-string key at {path}")
        return {
            key: _json_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, f"unsupported value at {path}")


def _coordinate_key(root: RootCoordinate) -> tuple[str, str, str]:
    return root.factor_type, root.scope_id, root.slot_id


def _coordinate_order(
    root: RootCoordinate, factor_ranks: Mapping[str, int]
) -> tuple[int, tuple[int, ...], int, bytes, bytes]:
    return (
        factor_ranks[root.factor_type],
        root.scope_order,
        root.slot_order,
        root.scope_id.encode("utf-8"),
        root.slot_id.encode("utf-8"),
    )


def _root_payload(root: RootCoordinate) -> dict[str, Any]:
    return {
        "factor_type": root.factor_type,
        "scope_id": root.scope_id,
        "slot_id": root.slot_id,
        "canonical_value": _json_value(root.canonical_value, path="root.canonical_value"),
    }


def _normalize_candidate(
    candidate: FactorizedCandidate, factor_ranks: Mapping[str, int]
) -> tuple[dict[str, Any], tuple[RootCoordinate, ...]]:
    if (
        not isinstance(candidate.candidate_id, str)
        or not candidate.candidate_id.strip()
        or not isinstance(candidate.roots, tuple)
    ):
        raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "candidate envelope")
    seen: set[tuple[str, str, str]] = set()
    ordered_roots: list[RootCoordinate] = []
    for root in candidate.roots:
        if not isinstance(root, RootCoordinate):
            raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "root type")
        if not isinstance(root.factor_type, str) or root.factor_type not in factor_ranks:
            raise CadenceGroupingError(CADENCE_FACTOR_TYPE_UNREGISTERED, str(root.factor_type))
        if (
            not isinstance(root.scope_id, str)
            or not root.scope_id
            or not isinstance(root.slot_id, str)
            or not root.slot_id
            or not isinstance(root.scope_order, tuple)
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in root.scope_order
            )
            or isinstance(root.slot_order, bool)
            or not isinstance(root.slot_order, int)
            or root.slot_order < 0
        ):
            raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "root coordinate")
        coordinate = _coordinate_key(root)
        if coordinate in seen:
            raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "duplicate root coordinate")
        seen.add(coordinate)
        ordered_roots.append(root)
    ordered_roots.sort(key=lambda item: _coordinate_order(item, factor_ranks))
    payload = {
        "kind": "CADENCE_CANDIDATE",
        "roots": [_root_payload(root) for root in ordered_roots],
        "derived": _json_value(candidate.derived, path="derived"),
        "residual": _json_value(candidate.residual, path="residual"),
    }
    try:
        canonical_bytes(payload)
    except (CanonicalSerializeError, TypeError, ValueError) as exc:
        raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "non-canonical payload") from exc
    return payload, tuple(ordered_roots)


def group_equivalent_candidates(
    candidates: Sequence[FactorizedCandidate],
) -> tuple[SemanticClass, ...]:
    """以完整 roots+derived+residual 表示分组，不把相同 roots 当作充分等价。"""

    if (
        isinstance(candidates, (str, bytes))
        or not isinstance(candidates, Sequence)
        or not candidates
    ):
        raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "candidate sequence")
    factor_ranks = {factor: index for index, factor in enumerate(CADENCE_FACTOR_ORDER)}
    seen_candidate_ids: set[str] = set()
    coordinate_scope_orders: dict[tuple[str, str, str], tuple[int, ...]] = {}
    slot_ids_by_namespace: dict[tuple[str, str], set[str]] = {}
    members: dict[bytes, tuple[dict[str, Any], tuple[RootCoordinate, ...], list[str]]] = {}
    for candidate in candidates:
        if not isinstance(candidate, FactorizedCandidate):
            raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "candidate type")
        if candidate.candidate_id in seen_candidate_ids:
            raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "duplicate candidate_id")
        seen_candidate_ids.add(candidate.candidate_id)
        payload, roots = _normalize_candidate(candidate, factor_ranks)
        for root in roots:
            coordinate = _coordinate_key(root)
            previous_scope_order = coordinate_scope_orders.setdefault(
                coordinate, root.scope_order
            )
            if previous_scope_order != root.scope_order:
                raise CadenceGroupingError(
                    CADENCE_FACTORIZATION_INVALID,
                    "root key has inconsistent query-scope order",
                )
            slot_ids_by_namespace.setdefault(
                (root.factor_type, root.scope_id), set()
            ).add(root.slot_id)
        identity_payload = dict(payload)
        identity_payload["roots"] = sorted(
            payload["roots"],
            key=lambda item: (
                factor_ranks[item["factor_type"]],
                item["scope_id"].encode("utf-8"),
                item["slot_id"].encode("utf-8"),
            ),
        )
        try:
            identity = canonical_bytes(identity_payload)
        except (CanonicalSerializeError, TypeError, ValueError) as exc:
            raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "candidate identity") from exc
        if identity not in members:
            members[identity] = (payload, roots, [])
        members[identity][2].append(candidate.candidate_id)

    # slot_order must depend on the semantic slot key alone. The SQL traversal can
    # create a different number of other roots before the same key in two candidates;
    # copying that incidental counter makes identical root keys fail alignment.
    slot_orders = {
        (factor_type, scope_id, slot_id): rank
        for (factor_type, scope_id), slot_ids in slot_ids_by_namespace.items()
        for rank, slot_id in enumerate(sorted(slot_ids, key=lambda item: item.encode("utf-8")))
    }
    classes: list[SemanticClass] = []
    for identity, (payload, roots, candidate_ids) in members.items():
        class_id = hashlib.sha256(identity).hexdigest()
        normalized_roots = tuple(
            sorted(
                (
                    replace(
                        root,
                        slot_order=slot_orders[
                            (root.factor_type, root.scope_id, root.slot_id)
                        ],
                    )
                    for root in roots
                ),
                key=lambda root: _coordinate_order(root, factor_ranks),
            )
        )
        classes.append(
            SemanticClass(
                class_id=class_id,
                candidate_ids=tuple(sorted(candidate_ids, key=lambda item: item.encode("utf-8"))),
                roots=normalized_roots,
                derived=payload["derived"],
                residual=payload["residual"],
            )
        )
    return tuple(sorted(classes, key=lambda item: item.class_id.encode("utf-8")))


def build_root_partitions(
    classes: Sequence[SemanticClass],
    *,
    active_class_ids: Sequence[str] | None = None,
) -> tuple[RootPartition, ...]:
    """按 root value 生成所有至少含两个非空 cell 的确定性分区。

    缺失坐标作为独立 ``ABSENT`` 值；无法规范化的输入须在 factorization 边界
    拒绝，不能把解析失败伪装成候选值。
    """

    factor_ranks = {factor: index for index, factor in enumerate(CADENCE_FACTOR_ORDER)}
    by_id = {item.class_id: item for item in classes}
    if len(by_id) != len(classes):
        raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "duplicate class_id")
    if active_class_ids is None:
        active = tuple(sorted(by_id, key=lambda item: item.encode("utf-8")))
    else:
        if (
            isinstance(active_class_ids, (str, bytes))
            or not isinstance(active_class_ids, Sequence)
            or any(not isinstance(item, str) or not item for item in active_class_ids)
            or len(set(active_class_ids)) != len(active_class_ids)
        ):
            raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "active class ids")
        active = tuple(sorted(active_class_ids, key=lambda item: item.encode("utf-8")))
        if not set(active).issubset(by_id):
            raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "unknown active class")
    if not active:
        raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "empty active view")

    roots_by_class: dict[str, dict[tuple[str, str, str], RootCoordinate]] = {}
    coordinate_metadata: dict[tuple[str, str, str], tuple[tuple[int, ...], int]] = {}
    for class_id in active:
        class_roots = by_id[class_id].roots
        root_map = {_coordinate_key(root): root for root in class_roots}
        if len(root_map) != len(class_roots):
            raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "duplicate class coordinate")
        roots_by_class[class_id] = root_map
        for coordinate, root in root_map.items():
            metadata = (root.scope_order, root.slot_order)
            previous = coordinate_metadata.setdefault(coordinate, metadata)
            if previous != metadata:
                raise CadenceGroupingError(
                    CADENCE_FACTORIZATION_INVALID, "coordinate order mismatch"
                )

    representatives = {
        coordinate: next(
            roots_by_class[class_id][coordinate]
            for class_id in active
            if coordinate in roots_by_class[class_id]
        )
        for coordinate in coordinate_metadata
    }
    coordinate_keys = sorted(
        coordinate_metadata,
        key=lambda coordinate: _coordinate_order(representatives[coordinate], factor_ranks),
    )
    partitions: list[RootPartition] = []
    for coordinate in coordinate_keys:
        roots_for_coordinate = [
            roots_by_class[class_id].get(coordinate) for class_id in active
        ]
        values_to_members: dict[bytes, tuple[dict[str, Any], list[str]]] = {}
        for class_id, root_value in zip(active, roots_for_coordinate, strict=True):
            value = (
                {"presence": "ABSENT"}
                if root_value is None
                else {
                    "presence": "PRESENT",
                    "canonical_value": root_value.canonical_value,
                }
            )
            normalized_value = _json_value(value, path="partition.value")
            try:
                key = canonical_bytes(normalized_value)
            except (CanonicalSerializeError, TypeError, ValueError) as exc:
                raise CadenceGroupingError(
                    CADENCE_FACTORIZATION_INVALID, "partition value"
                ) from exc
            if key not in values_to_members:
                values_to_members[key] = (normalized_value, [])
            values_to_members[key][1].append(class_id)
        if len(values_to_members) < 2:
            continue
        groups = tuple(
            tuple(sorted(members, key=lambda item: item.encode("utf-8")))
            for _key, (_value, members) in sorted(
                values_to_members.items(), key=lambda item: item[0]
            )
        )
        representative = next(root for root in roots_for_coordinate if root is not None)
        partition_payload = {
            "kind": "CADENCE_ROOT_PARTITION",
            "factor_type": representative.factor_type,
            "scope_id": representative.scope_id,
            "scope_order": [str(index) for index in representative.scope_order],
            "slot_id": representative.slot_id,
            "slot_order": str(representative.slot_order),
            "active_class_ids": list(active),
            "groups": [list(group) for group in groups],
        }
        try:
            partition_digest = hashlib.sha256(canonical_bytes(partition_payload)).hexdigest()
        except (CanonicalSerializeError, TypeError, ValueError) as exc:
            raise CadenceGroupingError(CADENCE_FACTORIZATION_INVALID, "partition identity") from exc
        partitions.append(
            RootPartition(
                partition_id=partition_digest,
                factor_type=representative.factor_type,
                scope_id=representative.scope_id,
                slot_id=representative.slot_id,
                query_scope_order=representative.scope_order,
                slot_order=representative.slot_order,
                groups=groups,
            )
        )
    return tuple(partitions)

