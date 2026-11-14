"""把受支持的 SQL query tree 因子化为 root、derived 和 residual。

超出表示范围的查询返回稳定 reason code，避免丢失的结构被合并进 ``ABSENT`` 或
residual，进而误判候选等价。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import sqlglot
from sqlglot import exp

from cadence.canonical_serialize import CanonicalSerializeError, canonical_bytes
from cadence.constants import (
    CADENCE_AGGREGATE_FUNCTIONS,
    CADENCE_DERIVED_ORDER,
    CADENCE_FACTOR_ORDER,
)
from cadence.grouping import FactorizedCandidate, RootCoordinate
from cadence.parse_bind import (
    QUERY_CONTRACT_CADENCE,
    BoundBetween,
    BoundBinary,
    BoundCast,
    BoundColumn,
    BoundDerivedColumn,
    BoundExpr,
    BoundFunc,
    BoundIn,
    BoundIsNull,
    BoundLiteral,
    BoundNotExistsCorrelated,
    BoundOrderItem,
    BoundOuterColumn,
    BoundScalarSubquery,
    BoundStar,
    BoundSubquery,
    BoundUnary,
    BoundUnsupported,
    BoundWindow,
    BoundWindowFrame,
    BoundWindowFrameBound,
    ParseBindError,
    SchemaCatalog,
    bind_sql,
)
from cadence.relational_algebra import (
    AlgebraQuery,
    Apply,
    FromSubquery,
    NoRelation,
    Relation,
    decompose_algebra_query,
    lower_to_extended_algebra,
)
from cadence.relational_algebra import (
    Join as AlgebraJoin,
)
from cadence.relational_algebra import (
    SetOperation as AlgebraSetOperation,
)

CADENCE_SQL_INPUT_INVALID = "CADENCE_SQL_INPUT_INVALID"
# Raw SQL / parser 子域、聚合注册、绑定节点、时间元数据和 canonical 编码均使用
# 独立 reason code；调用方可区分不支持与基础输入错误，禁止把失败当成 ABSENT。
CADENCE_SQL_BIND_REJECTED = "CADENCE_SQL_BIND_REJECTED"
CADENCE_SQL_UNSUPPORTED_QUERY_FEATURE = "CADENCE_SQL_UNSUPPORTED_QUERY_FEATURE"
CADENCE_SQL_AGGREGATE_UNREGISTERED = "CADENCE_SQL_AGGREGATE_UNREGISTERED"
CADENCE_SQL_UNSUPPORTED_BOUND_NODE = "CADENCE_SQL_UNSUPPORTED_BOUND_NODE"
CADENCE_SQL_TEMPORAL_COLUMN_INVALID = "CADENCE_SQL_TEMPORAL_COLUMN_INVALID"
CADENCE_SQL_FACTORIZATION_INVALID = "CADENCE_SQL_FACTORIZATION_INVALID"
CADENCE_SQL_CANONICALIZATION_FAILED = "CADENCE_SQL_CANONICALIZATION_FAILED"

_AGGREGATE_FUNCTIONS = CADENCE_AGGREGATE_FUNCTIONS
_SUPPORTED_SELECT_ARGUMENTS = frozenset(
    {
        "expressions",
        "from_",
        "joins",
        "where",
        "group",
        "having",
        "order",
        "limit",
        "offset",
        "with_",
        "distinct",
    }
)
_TEMPORAL_GROUP_FUNCTIONS = frozenset({"date_trunc", "timestamp_trunc"})
_TEMPORAL_GRANULARITIES = frozenset(
    {
        "microsecond",
        "millisecond",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "iso_week",
        "month",
        "quarter",
        "year",
        "decade",
        "century",
        "millennium",
        "dow",
        "isodow",
        "doy",
        "epoch",
    }
)


class CadenceSQLFactorizationError(ValueError):
    """SQL 因子化的稳定失败；调用端只依赖 ``reason_code``。"""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(reason_code if not detail else f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


@dataclass
class _ScopeContext:
    """一个 SQL query scope 的确定性坐标和已生成 root 索引。"""

    scope_id: str
    scope_order: tuple[int, ...]
    temporal_refs: dict[tuple[str, str], dict[str, str]] = field(default_factory=dict)
    slot_order: int = 0
    aggregate_cache: list[tuple[BoundFunc, dict[str, str]]] = field(default_factory=list)
    window_cache: list[tuple[BoundWindow, dict[str, str]]] = field(default_factory=list)
    temporal_group_cache: list[tuple[BoundFunc, dict[str, str]]] = field(default_factory=list)
    applications: dict[str, Apply] = field(default_factory=dict)
    window_expressions: dict[str, BoundWindow] = field(default_factory=dict)
    aggregate_expressions: frozenset[BoundFunc] = field(default_factory=frozenset)


class _CadenceFactorizer:
    """从扩展关系代数树提取 root、derived 与 residual。"""

    def __init__(
        self,
        *,
        candidate_id: str,
        temporal_columns: frozenset[tuple[str, str]],
        factor_order: tuple[str, ...],
        derived_order: tuple[str, ...],
    ) -> None:
        self._candidate_id = candidate_id
        self._temporal_columns = temporal_columns
        self._factor_ranks = {factor: i for i, factor in enumerate(factor_order)}
        self._derived_ranks = {derived: i for i, derived in enumerate(derived_order)}
        self._roots: list[RootCoordinate] = []
        self._derived: list[dict[str, Any]] = []
        self._scope_order: list[dict[str, str]] = []
        self._next_scope_order = 0
        self._root_keys: set[tuple[str, str, str]] = set()

    def factorize(self, query: AlgebraQuery) -> FactorizedCandidate:
        """将唯一顶层 query 变成候选因子化表示。"""

        if query.query_contract != QUERY_CONTRACT_CADENCE:
            raise CadenceSQLFactorizationError(
                CADENCE_SQL_BIND_REJECTED, "unexpected query contract"
            )
        skeleton = self._query(query, "q", None)
        ordered_roots = tuple(
            sorted(
                self._roots,
                key=lambda root: (
                    self._factor_ranks[root.factor_type],
                    root.scope_order,
                    root.slot_order,
                    root.scope_id.encode("utf-8"),
                    root.slot_id.encode("utf-8"),
                ),
            )
        )
        ordered_derived = tuple(
            sorted(
                self._derived,
                key=lambda item: (
                    self._derived_ranks[item["type"]],
                    str(item["site"]["scope_id"]).encode("utf-8"),
                    str(item["site"]["slot_id"]).encode("utf-8"),
                    canonical_bytes(item),
                ),
            )
        )
        residual = {
            "kind": "CADENCE_RESIDUAL",
            "scope_order": self._scope_order,
            "query_tree": skeleton,
        }
        try:
            # 在离开抽取边界前验证所有身份值均可由冻结序列化器编码；这可避免
            # 非有限数、浮点 token 或非字符串 key 直到实验落盘时才造成不稳定。
            canonical_bytes(
                {
                    "roots": [_root_identity(root) for root in ordered_roots],
                    "derived": list(ordered_derived),
                    "residual": residual,
                }
            )
        except (CanonicalSerializeError, TypeError, ValueError) as exc:
            raise CadenceSQLFactorizationError(
                CADENCE_SQL_CANONICALIZATION_FAILED, type(exc).__name__
            ) from exc
        return FactorizedCandidate(
            candidate_id=self._candidate_id,
            roots=ordered_roots,
            derived=ordered_derived,
            residual=residual,
        )

    def _query(
        self,
        query: AlgebraQuery,
        scope_id: str,
        parent_order: tuple[int, ...] | None,
    ) -> dict[str, Any]:
        """抽取一个 SELECT/query scope；子 scope 按 query tree 左到右深度优先编号。"""

        parts = decompose_algebra_query(query)
        ordinal = self._next_scope_order
        self._next_scope_order += 1
        scope_order = (ordinal,)
        context = _ScopeContext(
            scope_id=scope_id,
            scope_order=scope_order,
            applications=parts.applications,
            window_expressions={item.path: item.expression for item in parts.windows},
            aggregate_expressions=frozenset(parts.aggregate_references),
        )
        self._scope_order.append({"scope_id": scope_id, "ordinal": str(ordinal)})

        # 先收集本 scope 的显式时间列并注册 root；后续表达式统一引用该 root，
        # 避免 JOIN/投影/谓词遍历先后改变同一列的身份和值。
        for relation_id, column_id in sorted(
            self._scope_temporal_columns(query), key=_pair_order
        ):
            slot = _temporal_slot(relation_id, column_id)
            context.temporal_refs[(relation_id, column_id)] = self._add_root(
                context,
                factor_type="TIME_REFERENCE",
                slot_id=slot,
                value={"relation_id": relation_id, "column_id": column_id},
            )

        if parts.set_operation is not None:
            return self._set_operation(
                parts.set_operation, context, scope_id=scope_id, path="set/root"
            )

        # CTE definitions and derived sources are query scopes in their structural order.
        cte_refs: list[dict[str, str]] = []
        for index, cte in enumerate(query.ctes):
            slot = f"cte/{index}"
            child_scope = f"{scope_id}/cte/{index}"
            root_ref = self._subquery_root(
                context,
                slot=slot,
                role="CTE_DEFINITION",
                child_scope=child_scope,
            )
            self._query(cte, child_scope, scope_order)
            cte_refs.append(root_ref)

        from_source, from_skeleton = self._source(
            parts.source,
            context,
            scope_id=scope_id,
            slot="source/from",
            child_scope=f"{scope_id}/from",
        )

        join_values: list[dict[str, Any]] = []
        join_skeletons: list[dict[str, Any]] = []
        for index, join in enumerate(parts.joins):
            target, target_skeleton = self._join_source(
                join,
                context,
                scope_id=scope_id,
                index=index,
            )
            on_value = (
                self._expr(
                    join.predicate,
                    context,
                    path=f"join/{index}/on",
                    allow_temporal_group=False,
                )
                if join.predicate is not None
                else None
            )
            join_values.append(
                {"target": target, "join_type": join.join_type, "on": on_value}
            )
            join_skeletons.append(
                {
                    "ordinal": str(index),
                    "target": target_skeleton,
                    "join_type": join.join_type,
                    "on": on_value,
                }
            )

        # GROUP BY 先于 SELECT projection 遍历，确保已识别的 DATE_TRUNC 派生结构
        # 在投影/谓词重用时由同一 derived reference 指向，而不重复保存其值。
        group_refs: list[dict[str, str]] = []
        for index, expression in enumerate(parts.grouping_keys):
            slot = f"group_by/{index}"
            value = self._expr(
                expression,
                context,
                path=slot,
                allow_temporal_group=True,
            )
            root_ref = self._add_root(
                context,
                factor_type="GROUPING_KEY",
                slot_id=slot,
                value={"expression": value},
            )
            group_refs.append(root_ref)

        projection_exprs: list[BoundExpr] = []
        projection_refs: list[dict[str, str]] = []
        for index, projection in enumerate(parts.projections):
            projection_exprs.append(projection.expr)
            slot = f"projection/{index}"
            value = self._expr(
                projection.expr,
                context,
                path=slot,
                allow_temporal_group=False,
            )
            projection_refs.append(
                self._add_root(
                    context,
                    factor_type="RESULT_PROJECTION",
                    slot_id=slot,
                    value={"expression": value},
                )
            )

        predicate_skeleton: dict[str, dict[str, str]] = {}
        if parts.where is not None:
            slot = "predicate/where"
            value = self._expr(
                parts.where, context, path=slot, allow_temporal_group=False
            )
            root_ref = self._add_root(
                context,
                factor_type="PREDICATE",
                slot_id=slot,
                value={"clause": "WHERE", "expression": value},
            )
            predicate_skeleton["where"] = root_ref
        if parts.having is not None:
            slot = "predicate/having"
            value = self._expr(
                parts.having, context, path=slot, allow_temporal_group=False
            )
            root_ref = self._add_root(
                context,
                factor_type="PREDICATE",
                slot_id=slot,
                value={"clause": "HAVING", "expression": value},
            )
            predicate_skeleton["having"] = root_ref

        order_skeletons: list[dict[str, Any]] = []
        for index, item in enumerate(parts.order_by):
            slot = f"order_by/{index}"
            projection_matches = [
                projection_index
                for projection_index, projection_expr in enumerate(projection_exprs)
                if projection_expr == item.expr
            ]
            if len(projection_matches) == 1:
                projection_index = projection_matches[0]
                # 派生表结构不定义 ORDERING_TARGET_REFERENCE；相同投影目标必须以
                # RESULT_PROJECTION 根的符号引用表示，避免增加未支持的派生结构。
                expression_value: dict[str, Any] = projection_refs[projection_index]
            else:
                expression_value = self._expr(
                    item.expr,
                    context,
                    path=slot,
                    allow_temporal_group=True,
                )
            value = {
                "expression": expression_value,
                "direction": item.direction,
                "null_order": item.null_order,
            }
            order_ref = self._add_root(
                context,
                factor_type="ORDERING",
                slot_id=slot,
                value=value,
            )
            order_skeletons.append({"ordinal": str(index), "root": order_ref})

        source_root = self._source_relation_root(query, context)
        join_root: dict[str, str] | None = None
        if parts.joins:
            join_root = self._add_root(
                context,
                factor_type="JOIN_PATH",
                slot_id="join_path",
                value={"from": from_source, "joins": join_values},
            )
        if source_root is not None and join_root is not None:
            source_ref, source_ids = source_root
            join_ids = _direct_relation_ids(query)
            path_only = tuple(sorted(join_ids - source_ids, key=lambda x: x.encode("utf-8")))
            if path_only:
                self._add_derived(
                    context,
                    derived_type="JOIN_PATH_ONLY_RELATION",
                    slot_id="join_path/only_relations",
                    parents=(source_ref, join_root),
                    rule="JOIN_RELATIONS_MINUS_SOURCE_RELATIONS",
                    value={"relation_ids": list(path_only)},
                )
            source_value: dict[str, str] | None = source_ref
        else:
            source_value = source_root[0] if source_root is not None else None

        distinct_ref: dict[str, str] | None = None
        if parts.distinct:
            distinct_ref = self._add_root(
                context,
                factor_type="DUPLICATE_ELIMINATION",
                slot_id="query/distinct",
                value={"enabled": True},
            )

        limit_ref: dict[str, str] | None = None
        if parts.limit is not None or parts.offset is not None:
            limit_ref = self._add_root(
                context,
                factor_type="LIMIT_POLICY",
                slot_id="query/limit_policy",
                value={
                    "limit": str(parts.limit) if parts.limit is not None else None,
                    "offset": str(parts.offset) if parts.offset is not None else None,
                    "with_ties": parts.limit_with_ties,
                },
            )

        skeleton = {
            "kind": "SELECT",
            "scope_id": scope_id,
            "scope": query.scope,
            "ctes": cte_refs,
            "from": from_skeleton,
            "joins": join_skeletons,
            "source_relations": source_value,
            "join_path": join_root,
            "projections": [
                {"ordinal": str(i), "root": root}
                for i, root in enumerate(projection_refs)
            ],
            "where": predicate_skeleton.get("where"),
            "group_by": [
                {"ordinal": str(i), "root": ref}
                for i, ref in enumerate(group_refs)
            ],
            "having": predicate_skeleton.get("having"),
            "order_by": order_skeletons,
            "distinct": distinct_ref,
            "limit_policy": limit_ref,
        }
        return skeleton

    def _set_operation(
        self,
        node: AlgebraSetOperation,
        context: _ScopeContext,
        *,
        scope_id: str,
        path: str,
    ) -> dict[str, Any]:
        """记录每个有序 set node；左右 branch 各自成为 query scope。"""

        slot = f"{path}/composition"
        root = self._add_root(
            context,
            factor_type="SET_COMPOSITION",
            slot_id=slot,
            value={"operator": node.operator, "quantifier": node.quantifier},
        )
        left = self._set_branch(
            node.left,
            context,
            scope_id=scope_id,
            side="left",
            path=f"{path}/left",
        )
        right = self._set_branch(
            node.right,
            context,
            scope_id=scope_id,
            side="right",
            path=f"{path}/right",
        )
        return {"kind": "SET_OPERATION", "composition": root, "left": left, "right": right}

    def _set_branch(
        self,
        branch: AlgebraQuery | AlgebraSetOperation,
        context: _ScopeContext,
        *,
        scope_id: str,
        side: str,
        path: str,
    ) -> dict[str, Any]:
        if isinstance(branch, AlgebraQuery):
            child_scope = f"{scope_id}/set/{side}"
            root_ref = self._subquery_root(
                context,
                slot=f"{path}/branch",
                role=f"SET_{side.upper()}_BRANCH",
                child_scope=child_scope,
            )
            child_skeleton = self._query(branch, child_scope, context.scope_order)
            return {"scope_reference": root_ref, "query": child_skeleton}
        return self._set_operation(
            branch,
            context,
            scope_id=scope_id,
            path=path,
        )

    def _source(
        self,
        source: Relation | FromSubquery | NoRelation | None,
        context: _ScopeContext,
        *,
        scope_id: str,
        slot: str,
        child_scope: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if source is None or isinstance(source, NoRelation):
            return None, None
        if isinstance(source, Relation):
            value = {"kind": "RELATION", "relation_id": source.relation_id}
            return value, value
        if not isinstance(source, FromSubquery):
            raise CadenceSQLFactorizationError(CADENCE_SQL_UNSUPPORTED_BOUND_NODE, "source")
        root = self._subquery_root(
            context,
            slot=slot,
            role="FROM_DERIVED_QUERY",
            child_scope=child_scope,
        )
        child = self._query(source.query, child_scope, context.scope_order)
        return {"kind": "SUBQUERY_REFERENCE", "root": root}, {
            "kind": "SUBQUERY_REFERENCE",
            "root": root,
            "query": child,
        }

    def _join_source(
        self,
        join: AlgebraJoin,
        context: _ScopeContext,
        *,
        scope_id: str,
        index: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(join.target, Relation):
            value = {"kind": "RELATION", "relation_id": join.target.relation_id}
            return value, value
        if not isinstance(join.target, FromSubquery):
            raise CadenceSQLFactorizationError(CADENCE_SQL_UNSUPPORTED_BOUND_NODE, "join source")
        child_scope = f"{scope_id}/join/{index}"
        root = self._subquery_root(
            context,
            slot=f"join/{index}/source",
            role="JOIN_DERIVED_QUERY",
            child_scope=child_scope,
        )
        child = self._query(join.target.query, child_scope, context.scope_order)
        reference = {"kind": "SUBQUERY_REFERENCE", "root": root}
        return reference, {**reference, "query": child}

    def _source_relation_root(
        self, query: AlgebraQuery, context: _ScopeContext
    ) -> tuple[dict[str, str], set[str]] | None:
        """提取当前 scope 的语义来源端点，join-only 关系留给 derived 结构。"""

        direct_relations = _direct_relation_ids(query)
        if not direct_relations:
            return None

        referenced: set[str] = set()
        expressions = list(_direct_query_expressions(query))
        # JOIN predicates belong only to JOIN_PATH; they do not make an intermediate
        # relation a source endpoint by themselves.
        for expression in expressions:
            if expression is None:
                continue
            referenced.update(_bound_column_relations(expression, include_join_inner=False))
        referenced.intersection_update(direct_relations)
        if not decompose_algebra_query(query).joins or not referenced:
            referenced = direct_relations
        sorted_ids = tuple(sorted(referenced, key=lambda value: value.encode("utf-8")))
        root = self._add_root(
            context,
            factor_type="SOURCE_RELATION",
            slot_id="source/relations",
            value={"relation_ids": list(sorted_ids)},
        )
        return root, set(sorted_ids)

    def _scope_temporal_columns(self, query: AlgebraQuery) -> set[tuple[str, str]]:
        """Use explicit schema metadata or DATE_TRUNC operands; never infer by name."""

        expressions = _direct_query_expressions(query)
        mentioned = set().union(*(_bound_columns(item) for item in expressions))
        date_trunc = set().union(*(_date_trunc_columns(item) for item in expressions))
        return set(self._temporal_columns & mentioned) | date_trunc

    def _expr(
        self,
        expression: BoundExpr,
        context: _ScopeContext,
        *,
        path: str,
        allow_temporal_group: bool,
    ) -> dict[str, Any]:
        """规范化 bound expression，并提取嵌套 root/derived。"""

        if isinstance(expression, BoundColumn):
            key = (expression.relation_id, expression.column_id)
            if key in context.temporal_refs:
                return context.temporal_refs[key]
            return {
                "kind": "COLUMN",
                "relation_id": expression.relation_id,
                "column_id": expression.column_id,
            }
        if isinstance(expression, BoundDerivedColumn):
            return {
                "kind": "DERIVED_OUTPUT_SLOT",
                "scope_path": expression.scope_path,
                "projection_ordinal": str(expression.projection_ordinal),
            }
        if isinstance(expression, BoundOuterColumn):
            return {
                "kind": "OUTER_COLUMN",
                "scope_depth": str(expression.scope_depth),
                "relation_id": expression.relation_id,
                "column_id": expression.column_id,
                "scope_path": expression.scope_path,
                "projection_ordinal": (
                    str(expression.projection_ordinal)
                    if expression.projection_ordinal is not None
                    else None
                ),
            }
        if isinstance(expression, BoundLiteral):
            return {
                "kind": "LITERAL",
                "typed_tag": expression.typed_tag,
                "canonical": expression.canonical,
            }
        if isinstance(expression, BoundStar):
            return {"kind": "STAR"}
        if isinstance(expression, BoundFunc):
            if expression in context.aggregate_expressions:
                return self._aggregate_expression(expression, context, path=path)
            if expression.name in _TEMPORAL_GROUP_FUNCTIONS:
                granularity_expression, time_expression = _temporal_group_parts(expression)
                if (
                    not isinstance(granularity_expression, BoundLiteral)
                    or granularity_expression.typed_tag != "STRING"
                    or granularity_expression.canonical is None
                    or not isinstance(time_expression, BoundColumn)
                    or granularity_expression.canonical.casefold()
                    not in _TEMPORAL_GRANULARITIES
                ):
                    raise CadenceSQLFactorizationError(
                        CADENCE_SQL_UNSUPPORTED_QUERY_FEATURE,
                        "DATE_TRUNC requires a registered string unit and direct column",
                    )
            if (
                allow_temporal_group
                and expression.name in _TEMPORAL_GROUP_FUNCTIONS
                and len(expression.args) == 2
            ):
                return self._temporal_grouping(expression, context, path=path)
            cached_group = self._cached_temporal_group(expression, context)
            if cached_group is not None:
                return cached_group
            return {
                "kind": "FUNCTION",
                "name": expression.name,
                "distinct": expression.distinct,
                "arguments": [
                    self._expr(
                        argument,
                        context,
                        path=f"{path}/arg/{index}",
                        allow_temporal_group=False,
                    )
                    for index, argument in enumerate(expression.args)
                ],
            }
        if isinstance(expression, BoundWindow):
            expression = context.window_expressions.get(path, expression)
            cached = self._cached_window(expression, context)
            if cached is not None:
                return cached
            slot = f"window/{path}"
            function = self._expr(
                expression.function,
                context,
                path=f"{path}/function",
                allow_temporal_group=False,
            )
            value = {
                "function": function,
                "partition_by": [
                    self._expr(
                        item,
                        context,
                        path=f"{path}/partition/{index}",
                        allow_temporal_group=False,
                    )
                    for index, item in enumerate(expression.partition_by)
                ],
                "order_by": [
                    self._order_item(item, context, path=f"{path}/order/{index}")
                    for index, item in enumerate(expression.order_by)
                ],
                "frame": self._window_frame_value(
                    expression.frame,
                    context,
                    path=f"{path}/frame",
                ),
            }
            root = self._add_root(
                context,
                factor_type="WINDOW_SPECIFICATION",
                slot_id=slot,
                value=value,
            )
            context.window_cache.append((expression, root))
            return root
        if isinstance(expression, BoundCast):
            return {
                "kind": "CAST",
                "target_type": expression.target_type,
                "operand": self._expr(
                    expression.operand,
                    context,
                    path=f"{path}/operand",
                    allow_temporal_group=False,
                ),
            }
        if isinstance(expression, BoundBinary):
            return {
                "kind": "BINARY",
                "operator": expression.op,
                "left": self._expr(
                    expression.left,
                    context,
                    path=f"{path}/left",
                    allow_temporal_group=False,
                ),
                "right": self._expr(
                    expression.right,
                    context,
                    path=f"{path}/right",
                    allow_temporal_group=False,
                ),
            }
        if isinstance(expression, BoundUnary):
            return {
                "kind": "UNARY",
                "operator": expression.op,
                "operand": self._expr(
                    expression.operand,
                    context,
                    path=f"{path}/operand",
                    allow_temporal_group=False,
                ),
            }
        if isinstance(expression, BoundBetween):
            return {
                "kind": "BETWEEN",
                "operand": self._expr(
                    expression.operand,
                    context,
                    path=f"{path}/operand",
                    allow_temporal_group=False,
                ),
                "lower": self._expr(
                    expression.lower,
                    context,
                    path=f"{path}/lower",
                    allow_temporal_group=False,
                ),
                "upper": self._expr(
                    expression.upper,
                    context,
                    path=f"{path}/upper",
                    allow_temporal_group=False,
                ),
                "negated": expression.negated,
            }
        if isinstance(expression, BoundIn):
            if expression.subquery is not None:
                slot = f"subquery/{path}"
                child_scope = f"{context.scope_id}/{path}/subquery"
                ref = self._subquery_root(
                    context,
                    slot=slot,
                    role="NOT_IN_QUERY" if expression.negated else "IN_QUERY",
                    child_scope=child_scope,
                )
                application = context.applications.get(path)
                if application is None or not isinstance(application.right, AlgebraQuery):
                    raise CadenceSQLFactorizationError(
                        CADENCE_SQL_UNSUPPORTED_BOUND_NODE, "missing IN Apply node"
                    )
                self._query(application.right, child_scope, context.scope_order)
                source: dict[str, Any] = {"query_reference": ref}
            else:
                source = {
                    "values": [
                        self._expr(
                            item,
                            context,
                            path=f"{path}/value/{index}",
                            allow_temporal_group=False,
                        )
                        for index, item in enumerate(expression.values)
                    ]
                }
            return {
                "kind": "IN",
                "operand": self._expr(
                    expression.operand,
                    context,
                    path=f"{path}/operand",
                    allow_temporal_group=False,
                ),
                "negated": expression.negated,
                **source,
            }
        if isinstance(expression, BoundIsNull):
            return {
                "kind": "IS_NULL",
                "operand": self._expr(
                    expression.operand,
                    context,
                    path=f"{path}/operand",
                    allow_temporal_group=False,
                ),
                "negated": expression.negated,
            }
        if isinstance(expression, BoundScalarSubquery):
            slot = f"subquery/{path}"
            child_scope = f"{context.scope_id}/{path}/subquery"
            ref = self._subquery_root(
                context,
                slot=slot,
                role="SCALAR_QUERY",
                child_scope=child_scope,
            )
            application = context.applications.get(path)
            if application is None or not isinstance(application.right, AlgebraQuery):
                raise CadenceSQLFactorizationError(
                    CADENCE_SQL_UNSUPPORTED_BOUND_NODE, "missing scalar Apply node"
                )
            self._query(application.right, child_scope, context.scope_order)
            return ref
        if isinstance(expression, BoundSubquery):
            slot = f"subquery/{path}"
            child_scope = f"{context.scope_id}/{path}/subquery"
            ref = self._subquery_root(
                context,
                slot=slot,
                role="SUBQUERY",
                child_scope=child_scope,
            )
            application = context.applications.get(path)
            if application is None or not isinstance(application.right, AlgebraQuery):
                raise CadenceSQLFactorizationError(
                    CADENCE_SQL_UNSUPPORTED_BOUND_NODE, "missing subquery Apply node"
                )
            self._query(application.right, child_scope, context.scope_order)
            return ref
        if isinstance(expression, BoundNotExistsCorrelated):
            slot = f"subquery/{path}"
            application = context.applications.get(path)
            relation_id = expression.source_table.relation_id
            inner_column = expression.inner_column
            outer_column = expression.outer_column
            if application is not None:
                if isinstance(application.right, Relation):
                    relation_id = application.right.relation_id
                if application.correlations:
                    correlation = application.correlations[0]
                    inner_column = BoundColumn(
                        correlation.inner_relation_id or inner_column.relation_id,
                        correlation.inner_column_id or inner_column.column_id,
                    )
                    outer_column = BoundColumn(
                        correlation.relation_id or outer_column.relation_id,
                        correlation.column_id or outer_column.column_id,
                    )
            return self._add_root(
                context,
                factor_type="SUBQUERY_STRUCTURE",
                slot_id=slot,
                value={
                    "role": "NOT_EXISTS_CORRELATED",
                    "source_relation": relation_id,
                    "inner_column": _column_value(inner_column),
                    "outer_column": _column_value(outer_column),
                },
            )
        if isinstance(expression, BoundUnsupported):
            raise CadenceSQLFactorizationError(
                CADENCE_SQL_UNSUPPORTED_BOUND_NODE, expression.construct
            )
        raise CadenceSQLFactorizationError(
            CADENCE_SQL_UNSUPPORTED_BOUND_NODE, type(expression).__name__
        )

    def _aggregate_expression(
        self,
        expression: BoundFunc,
        context: _ScopeContext,
        *,
        path: str,
    ) -> dict[str, str]:
        for previous, ref in context.aggregate_cache:
            if previous == expression:
                return ref
        slot = f"aggregate/{path}"
        aggregation_ref = self._add_root(
            context,
            factor_type="AGGREGATION",
            slot_id=slot,
            value={"function": expression.name, "distinct": expression.distinct},
        )
        metric = [
            self._expr(
                argument,
                context,
                path=f"{path}/metric/{index}",
                allow_temporal_group=False,
            )
            for index, argument in enumerate(expression.args)
        ]
        metric_ref = self._add_root(
            context,
            factor_type="METRIC_DEFINITION",
            slot_id=slot,
            value={"arguments": metric},
        )
        derived_ref = self._add_derived(
            context,
            derived_type="AGGREGATE_EXPRESSION",
            slot_id=slot,
            parents=(aggregation_ref, metric_ref),
            rule="APPLY_AGGREGATION_TO_METRIC_ARGUMENTS",
            value={
                "aggregation_reference": aggregation_ref,
                "metric_reference": metric_ref,
            },
        )
        context.aggregate_cache.append((expression, derived_ref))
        return derived_ref

    def _temporal_grouping(
        self,
        expression: BoundFunc,
        context: _ScopeContext,
        *,
        path: str,
    ) -> dict[str, str]:
        for previous, ref in context.temporal_group_cache:
            if previous == expression:
                return ref
        granularity_expression, time_expression = _temporal_group_parts(expression)
        if not isinstance(time_expression, BoundColumn):
            raise CadenceSQLFactorizationError(
                CADENCE_SQL_UNSUPPORTED_QUERY_FEATURE, "DATE_TRUNC time operand must be a column"
            )
        temporal_ref = context.temporal_refs.get(
            (time_expression.relation_id, time_expression.column_id)
        )
        if temporal_ref is None:
            raise CadenceSQLFactorizationError(
                CADENCE_SQL_FACTORIZATION_INVALID, "DATE_TRUNC time reference"
            )
        granularity_slot = f"{path}/granularity"
        granularity_value = self._expr(
            granularity_expression,
            context,
            path=granularity_slot,
            allow_temporal_group=False,
        )
        if (
            isinstance(granularity_expression, BoundLiteral)
            and granularity_expression.typed_tag == "STRING"
            and granularity_expression.canonical is not None
        ):
            granularity_value = {
                "kind": "LITERAL",
                "typed_tag": "STRING",
                "canonical": granularity_expression.canonical.casefold(),
            }
        granularity_ref = self._add_root(
            context,
            factor_type="GROUPING_GRANULARITY",
            slot_id=granularity_slot,
            value={"granularity": granularity_value},
        )
        time_refs = (temporal_ref,)
        parent_refs = (granularity_ref, temporal_ref)
        derived_ref = self._add_derived(
            context,
            derived_type="TEMPORAL_GROUPING_EXPRESSION",
            slot_id=path,
            parents=parent_refs,
            rule="DATE_TRUNC(GROUPING_GRANULARITY, TIME_REFERENCE)",
            value={
                "time_references": list(time_refs),
                "granularity_reference": granularity_ref,
            },
        )
        context.temporal_group_cache.append((expression, derived_ref))
        return derived_ref

    def _order_item(
        self,
        item: BoundOrderItem,
        context: _ScopeContext,
        *,
        path: str,
    ) -> dict[str, Any]:
        return {
            "expression": self._expr(
                item.expr,
                context,
                path=f"{path}/expression",
                allow_temporal_group=False,
            ),
            "direction": item.direction,
            "null_order": item.null_order,
        }

    def _add_root(
        self,
        context: _ScopeContext,
        *,
        factor_type: str,
        slot_id: str,
        value: Any,
    ) -> dict[str, str]:
        if factor_type not in self._factor_ranks or not slot_id:
            raise CadenceSQLFactorizationError(CADENCE_SQL_FACTORIZATION_INVALID, "root key")
        key = (factor_type, context.scope_id, slot_id)
        if key in self._root_keys:
            raise CadenceSQLFactorizationError(
                CADENCE_SQL_FACTORIZATION_INVALID, f"duplicate root slot {factor_type}:{slot_id}"
            )
        self._root_keys.add(key)
        root = RootCoordinate(
            factor_type=factor_type,
            scope_id=context.scope_id,
            scope_order=context.scope_order,
            slot_id=slot_id,
            slot_order=context.slot_order,
            canonical_value=value,
        )
        context.slot_order += 1
        self._roots.append(root)
        return _root_ref(factor_type, context.scope_id, slot_id)

    def _add_derived(
        self,
        context: _ScopeContext,
        *,
        derived_type: str,
        slot_id: str,
        parents: Sequence[dict[str, str]],
        rule: str,
        value: Any,
    ) -> dict[str, str]:
        if not parents:
            raise CadenceSQLFactorizationError(
                CADENCE_SQL_FACTORIZATION_INVALID, "derived parents"
            )
        self._derived.append(
            {
                "type": derived_type,
                "site": {"scope_id": context.scope_id, "slot_id": slot_id},
                "parent_roots": list(parents),
                "construction_rule": rule,
                "canonical_value": value,
            }
        )
        return _derived_ref(derived_type, context.scope_id, slot_id)

    def _cached_temporal_group(
        self, expression: BoundFunc, context: _ScopeContext
    ) -> dict[str, str] | None:
        """相同 DATE_TRUNC 只生成一份 granularity 与 derived identity。"""

        for previous, reference in context.temporal_group_cache:
            if previous == expression:
                return reference
        return None

    def _cached_window(
        self, expression: BoundWindow, context: _ScopeContext
    ) -> dict[str, str] | None:
        """同一 scope 中重复出现的窗口表达式共用一个 WINDOW_SPECIFICATION。"""

        for previous, reference in context.window_cache:
            if previous == expression:
                return reference
        return None

    def _window_frame_value(
        self,
        frame: BoundWindowFrame | None,
        context: _ScopeContext,
        *,
        path: str,
    ) -> dict[str, Any] | None:
        """保留窗口 root 的 frame 单位、边界、offset 和排除规则。"""

        if frame is None:
            return None

        def bound_value(
            bound: BoundWindowFrameBound,
            *,
            boundary_path: str,
        ) -> dict[str, Any]:
            if bound.kind == "UNBOUNDED":
                return {"kind": bound.kind, "side": bound.side}
            if bound.kind == "CURRENT_ROW":
                return {"kind": bound.kind}
            if bound.kind == "OFFSET" and bound.offset is not None:
                return {
                    "kind": bound.kind,
                    "side": bound.side,
                    "expression": self._expr(
                        bound.offset,
                        context,
                        path=f"{boundary_path}/offset",
                        allow_temporal_group=False,
                    ),
                }
            raise CadenceSQLFactorizationError(
                CADENCE_SQL_FACTORIZATION_INVALID,
                f"window frame bound at {boundary_path}",
            )

        return {
            "unit": frame.unit,
            "start": bound_value(frame.start, boundary_path=f"{path}/start"),
            "end": bound_value(frame.end, boundary_path=f"{path}/end"),
            "exclusion": frame.exclusion,
        }

    def _subquery_root(
        self,
        context: _ScopeContext,
        *,
        slot: str,
        role: str,
        child_scope: str,
    ) -> dict[str, str]:
        return self._add_root(
            context,
            factor_type="SUBQUERY_STRUCTURE",
            slot_id=f"subquery/{slot}",
            value={"role": role, "child_scope_id": child_scope},
        )


def factorize_sql(
    *,
    sql: str,
    candidate_id: str,
    catalog: SchemaCatalog,
    dialect: str,
    temporal_columns: Sequence[tuple[str, str]] | None = None,
) -> FactorizedCandidate:
    """解析 SQL 并生成 CADENCE 因子；方言和时间列 metadata 都显式输入。

    ``temporal_columns`` 使用 catalog 中的 canonical ``(relation_id, column_id)``；
    未提供类型信息时不按列名猜测时间列，只有显式 ``DATE_TRUNC`` 语境会产生
    ``TIME_REFERENCE``。解析固定使用独立的 CADENCE 查询范围，
    复用只读 SQL parser/binder，不调用通用查询语义投影。
    """

    if temporal_columns is None:
        raise CadenceSQLFactorizationError(
            CADENCE_SQL_TEMPORAL_COLUMN_INVALID,
            "caller must explicitly provide temporal-column metadata, including an empty set",
        )
    if (
        not isinstance(sql, str)
        or not sql.strip()
        or not isinstance(candidate_id, str)
        or not candidate_id
        or not isinstance(catalog, SchemaCatalog)
        or not isinstance(dialect, str)
        or not dialect
        or isinstance(temporal_columns, (str, bytes))
    ):
        raise CadenceSQLFactorizationError(CADENCE_SQL_INPUT_INVALID)
    temporal_set: set[tuple[str, str]] = set()
    try:
        for pair in temporal_columns:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(not isinstance(item, str) or not item for item in pair)
            ):
                raise CadenceSQLFactorizationError(CADENCE_SQL_TEMPORAL_COLUMN_INVALID)
            relation_id, column_id = pair
            relation = catalog.relations.get(relation_id)
            if relation is None or column_id not in relation.columns:
                raise CadenceSQLFactorizationError(CADENCE_SQL_TEMPORAL_COLUMN_INVALID)
            temporal_set.add((relation_id, column_id))
    except TypeError as exc:
        raise CadenceSQLFactorizationError(CADENCE_SQL_TEMPORAL_COLUMN_INVALID) from exc

    try:
        # 先检查原始 AST：绑定 IR 不保留 FETCH/WITH TIES 等部分语法，事后无法
        # 判断它们是否已被错误丢弃，所以 unsupported 必须在绑定前稳定拒绝。
        _validate_raw_query_features(sql, dialect)
        _validate_aggregate_registry(sql, dialect)
        bound = bind_sql(
            sql,
            catalog,
            dialect=dialect,
            query_contract=QUERY_CONTRACT_CADENCE,
        )
    except ParseBindError as exc:
        raise CadenceSQLFactorizationError(
            CADENCE_SQL_BIND_REJECTED, exc.reason_code
        ) from exc
    return _CadenceFactorizer(
        candidate_id=candidate_id,
        temporal_columns=frozenset(temporal_set),
        factor_order=CADENCE_FACTOR_ORDER,
        derived_order=CADENCE_DERIVED_ORDER,
    ).factorize(lower_to_extended_algebra(bound))


def _validate_raw_query_features(sql: str, dialect: str) -> None:
    """在绑定前拒绝 raw AST 中无法无损表示的子域与修饰。"""

    try:
        statements = sqlglot.parse(sql, read=dialect)
    except Exception as exc:  # noqa: BLE001 - 收敛 sqlglot 方言异常
        raise CadenceSQLFactorizationError(CADENCE_SQL_BIND_REJECTED, "SQL parse") from exc
    if len(statements) != 1 or statements[0] is None:
        raise CadenceSQLFactorizationError(CADENCE_SQL_UNSUPPORTED_QUERY_FEATURE, "statement count")
    for select in statements[0].find_all(exp.Select):
        for key, value in select.args.items():
            if value is not None and key not in _SUPPORTED_SELECT_ARGUMENTS:
                raise CadenceSQLFactorizationError(
                    CADENCE_SQL_UNSUPPORTED_QUERY_FEATURE, f"SELECT.{key}"
                )
        distinct = select.args.get("distinct")
        if isinstance(distinct, exp.Distinct) and distinct.args.get("on") is not None:
            raise CadenceSQLFactorizationError(
                CADENCE_SQL_UNSUPPORTED_QUERY_FEATURE, "DISTINCT ON"
            )
        limit_node = select.args.get("limit")
        if isinstance(limit_node, exp.Limit):
            options = limit_node.args.get("limit_options")
            if options is not None:
                raise CadenceSQLFactorizationError(
                    CADENCE_SQL_UNSUPPORTED_QUERY_FEATURE, "LIMIT modifier"
                )
        if isinstance(limit_node, exp.Fetch):
            options = limit_node.args.get("limit_options")
            if (
                limit_node.args.get("direction") not in {"FIRST", "NEXT"}
                or not isinstance(options, exp.LimitOptions)
                or options.args.get("rows") is not True
                or options.args.get("percent") is True
                or limit_node.args.get("count") is None
                or (options.args.get("with_ties") is True and select.args.get("order") is None)
            ):
                raise CadenceSQLFactorizationError(
                    CADENCE_SQL_UNSUPPORTED_QUERY_FEATURE, "FETCH shape"
                )


def _validate_aggregate_registry(sql: str, dialect: str) -> None:
    """只拆分规格登记的聚合函数；SQLGlot 已识别但未登记的聚合 fail-closed。"""

    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception as exc:  # noqa: BLE001 - bind_sql 已先作完整错误分类
        raise CadenceSQLFactorizationError(CADENCE_SQL_BIND_REJECTED, "SQL parse") from exc
    for node in tree.walk():
        if isinstance(node, exp.AggFunc):
            name = str(node.sql_name()).casefold()  # type: ignore[no-untyped-call]
            if name not in _AGGREGATE_FUNCTIONS:
                raise CadenceSQLFactorizationError(
                    CADENCE_SQL_AGGREGATE_UNREGISTERED, name
                )


def _direct_query_expressions(query: AlgebraQuery) -> tuple[BoundExpr, ...]:
    """返回本 scope 的投影/过滤/分组/排序表达式，不把 JOIN ON 越界成源引用。"""

    parts = decompose_algebra_query(query)
    expressions: list[BoundExpr] = [projection.expr for projection in parts.projections]
    expressions.extend(parts.grouping_keys)
    expressions.extend(item.expr for item in parts.order_by)
    if parts.where is not None:
        expressions.append(parts.where)
    if parts.having is not None:
        expressions.append(parts.having)
    return tuple(expressions)


def _date_trunc_columns(expression: BoundExpr) -> set[tuple[str, str]]:
    """找出当前 scope 中受支持 DATE_TRUNC 的直接列操作数。"""

    out: set[tuple[str, str]] = set()
    if (
        isinstance(expression, BoundFunc)
        and expression.name in _TEMPORAL_GROUP_FUNCTIONS
        and len(expression.args) == 2
    ):
        _, time_expression = _temporal_group_parts(expression)
        out.update(_bound_columns(time_expression))
    for child in _expr_children(expression):
        out.update(_date_trunc_columns(child))
    return out


def _temporal_group_parts(expression: BoundFunc) -> tuple[BoundExpr, BoundExpr]:
    """把方言函数参数规范成 (granularity, time expression)。"""

    if len(expression.args) != 2 or expression.name not in _TEMPORAL_GROUP_FUNCTIONS:
        raise CadenceSQLFactorizationError(
            CADENCE_SQL_UNSUPPORTED_BOUND_NODE, "DATE_TRUNC argument shape"
        )
    if expression.name == "timestamp_trunc":
        time_expression, granularity = expression.args
        return granularity, time_expression
    granularity, time_expression = expression.args
    return granularity, time_expression


def _direct_relation_ids(query: AlgebraQuery) -> set[str]:
    """返回当前 FROM/JOIN 中的直接基础关系，不递归穿透子查询 scope。"""

    relation_ids: set[str] = set()
    parts = decompose_algebra_query(query)
    if isinstance(parts.source, Relation):
        relation_ids.add(parts.source.relation_id)
    for join in parts.joins:
        if isinstance(join.target, Relation):
            relation_ids.add(join.target.relation_id)
    return relation_ids


def _bound_columns(expression: BoundExpr) -> set[tuple[str, str]]:
    """只沿当前表达式树收集绑定列，子查询 lineage 保持独立 scope。"""

    out: set[tuple[str, str]] = set()
    if isinstance(expression, BoundColumn):
        out.add((expression.relation_id, expression.column_id))
    else:
        for child in _expr_children(expression):
            out.update(_bound_columns(child))
    return out


def _bound_column_relations(
    expression: BoundExpr, *, include_join_inner: bool
) -> set[str]:
    """收集表达式直接引用的 relation ID，供 source endpoint 判定使用。"""

    out: set[str] = set()
    if isinstance(expression, BoundColumn):
        out.add(expression.relation_id)
    elif isinstance(expression, BoundNotExistsCorrelated):
        out.add(expression.outer_column.relation_id)
        if include_join_inner:
            out.add(expression.inner_column.relation_id)
    else:
        for child in _expr_children(expression):
            out.update(_bound_column_relations(child, include_join_inner=include_join_inner))
    return out


def _expr_children(expression: BoundExpr) -> tuple[BoundExpr, ...]:
    """以明确的 IR 子节点顺序遍历表达式，不用反射吞掉未知结构。"""

    if isinstance(expression, BoundFunc):
        return expression.args
    if isinstance(expression, BoundWindow):
        return (
            expression.function,
            *expression.partition_by,
            *(item.expr for item in expression.order_by),
        )
    if isinstance(expression, BoundCast):
        return (expression.operand,)
    if isinstance(expression, BoundBinary):
        return (expression.left, expression.right)
    if isinstance(expression, BoundUnary):
        return (expression.operand,)
    if isinstance(expression, BoundBetween):
        return (expression.operand, expression.lower, expression.upper)
    if isinstance(expression, BoundIn):
        return (expression.operand, *expression.values)
    if isinstance(expression, BoundIsNull):
        return (expression.operand,)
    return ()


def _column_value(column: BoundColumn) -> dict[str, str]:
    """生成不含显示名称的 canonical column identity。"""

    return {"relation_id": column.relation_id, "column_id": column.column_id}


def _temporal_slot(relation_id: str, column_id: str) -> str:
    """百分号编码 ID 中的分隔符，避免 relation/column 路径发生碰撞。"""

    return f"column/{quote(relation_id, safe='')}/{quote(column_id, safe='')}"


def _pair_order(pair: tuple[str, str]) -> tuple[bytes, bytes]:
    """用 UTF-8 字节序排序 schema pair，避免依赖 locale。"""

    return pair[0].encode("utf-8"), pair[1].encode("utf-8")


def _root_ref(factor_type: str, scope_id: str, slot_id: str) -> dict[str, str]:
    """生成对 root identity key 的封闭符号引用，不复制 root 值。"""

    return {
        "kind": "ROOT_REFERENCE",
        "factor_type": factor_type,
        "scope_id": scope_id,
        "slot_id": slot_id,
    }


def _derived_ref(derived_type: str, scope_id: str, slot_id: str) -> dict[str, str]:
    """生成对 derived structure 的封闭引用。"""

    return {
        "kind": "DERIVED_REFERENCE",
        "derived_type": derived_type,
        "scope_id": scope_id,
        "slot_id": slot_id,
    }


def _root_identity(root: RootCoordinate) -> dict[str, Any]:
    """把坐标序号转为 canonical string token，避免 identity 中出现数字 token。"""

    return {
        "factor_type": root.factor_type,
        "scope_id": root.scope_id,
        "scope_order": [str(item) for item in root.scope_order],
        "slot_id": root.slot_id,
        "slot_order": str(root.slot_order),
        "canonical_value": root.canonical_value,
    }



