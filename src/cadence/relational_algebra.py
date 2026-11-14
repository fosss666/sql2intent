"""将绑定 SQL 查询作用域降为扩展关系代数树。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from cadence.constants import CADENCE_AGGREGATE_FUNCTIONS
from cadence.parse_bind import (
    BoundBetween,
    BoundBinary,
    BoundCast,
    BoundDerivedTable,
    BoundExpr,
    BoundFunc,
    BoundIn,
    BoundIsNull,
    BoundJoin,
    BoundNotExistsCorrelated,
    BoundOrderItem,
    BoundOuterColumn,
    BoundProjection,
    BoundQuery,
    BoundScalarSubquery,
    BoundSetOperation,
    BoundSubquery,
    BoundTable,
    BoundUnary,
    BoundWindow,
    BoundWindowFrame,
    BoundWindowFrameBound,
)


class AlgebraNode:
    """关系代数树节点的共同基类。"""


@dataclass(frozen=True)
class Relation(AlgebraNode):
    """基础关系或视图。"""

    relation_id: str
    alias: str | None


@dataclass(frozen=True)
class NoRelation(AlgebraNode):
    """SELECT 不含 FROM 时的空输入。"""


@dataclass(frozen=True)
class FromSubquery(AlgebraNode):
    """FROM/JOIN 中作为关系来源的嵌套查询。"""

    query: AlgebraQuery
    alias: str | None
    output_names: tuple[str | None, ...]


@dataclass(frozen=True)
class Join(AlgebraNode):
    """有序二元连接；多表连接按 SQL 顺序组织为左深树。"""

    input: AlgebraNode
    target: Relation | FromSubquery
    join_type: str
    predicate: BoundExpr | None


@dataclass(frozen=True)
class Selection(AlgebraNode):
    """WHERE 或 HAVING 选择，保留原始绑定谓词及其三值语义。"""

    input: AlgebraNode
    clause: Literal["WHERE", "HAVING"]
    predicate: BoundExpr


@dataclass(frozen=True)
class GroupAggregate(AlgebraNode):
    """分组键与聚合表达式；空分组键保留全表聚合语义。"""

    input: AlgebraNode
    grouping_keys: tuple[BoundExpr, ...]
    aggregates: tuple[BoundFunc, ...]


@dataclass(frozen=True)
class WindowExpression:
    """窗口表达式在其所属 SQL 子句中的稳定路径。"""

    path: str
    expression: BoundWindow


@dataclass(frozen=True)
class Window(AlgebraNode):
    """窗口计算及其 partition/order/frame 绑定表达式。"""

    input: AlgebraNode
    expressions: tuple[WindowExpression, ...]


@dataclass(frozen=True)
class Projection(AlgebraNode):
    """按 SQL 顺序排列的输出表达式。"""

    input: AlgebraNode
    items: tuple[BoundProjection, ...]


@dataclass(frozen=True)
class DuplicateElimination(AlgebraNode):
    """DISTINCT 的重复行消除操作。"""

    input: AlgebraNode


@dataclass(frozen=True)
class Sort(AlgebraNode):
    """保留方向和 NULL 排序位置的有序排序键。"""

    input: AlgebraNode
    items: tuple[BoundOrderItem, ...]


@dataclass(frozen=True)
class Limit(AlgebraNode):
    """行数边界、偏移量及 WITH TIES 语义。"""

    input: AlgebraNode
    bound: int | None
    offset: int | None
    with_ties: bool


@dataclass(frozen=True)
class CorrelatedReference:
    """Apply 节点记录的外层引用或受限 NOT EXISTS 相关键。"""

    scope_depth: int
    relation_id: str | None
    column_id: str | None
    scope_path: str | None
    projection_ordinal: int | None
    inner_relation_id: str | None = None
    inner_column_id: str | None = None


@dataclass(frozen=True)
class Apply(AlgebraNode):
    """谓词或标量子查询的应用点，site 指明其出现的表达式位置。"""

    input: AlgebraNode
    right: AlgebraQuery | Relation
    role: str
    site: str
    correlations: tuple[CorrelatedReference, ...]


@dataclass(frozen=True)
class SetOperation(AlgebraNode):
    """保留 operator、ALL/DISTINCT 与左右分支次序的集合运算。"""

    operator: Literal["UNION", "INTERSECT", "EXCEPT"]
    quantifier: Literal["DISTINCT", "ALL"]
    left: AlgebraQuery | SetOperation
    right: AlgebraQuery | SetOperation


@dataclass(frozen=True)
class AlgebraQuery:
    """一个查询作用域的 bag 代数树和独立 CTE 定义；无 DISTINCT 时保留行重复。"""

    root: AlgebraNode
    ctes: tuple[AlgebraQuery, ...]
    scope: Literal["GLOBAL", "LOCAL"]
    query_contract: str


@dataclass(frozen=True)
class AlgebraQueryParts:
    """从代数树读出的 factorizer 输入；不重新解析 SQL 子句。"""

    source: Relation | FromSubquery | NoRelation | None
    joins: tuple[Join, ...]
    where: BoundExpr | None
    grouping_keys: tuple[BoundExpr, ...]
    aggregates: tuple[BoundFunc, ...]
    aggregate_references: tuple[BoundFunc, ...]
    having: BoundExpr | None
    windows: tuple[WindowExpression, ...]
    projections: tuple[BoundProjection, ...]
    distinct: bool
    order_by: tuple[BoundOrderItem, ...]
    limit: int | None
    offset: int | None
    limit_with_ties: bool
    set_operation: SetOperation | None
    applications: dict[str, Apply]


def lower_to_extended_algebra(query: BoundQuery) -> AlgebraQuery:
    """将绑定查询转换为代数树，保持表达式、作用域和 SQL 子句语义。"""

    if query.compound is not None:
        root: AlgebraNode = _lower_set_operation(query.compound)
    else:
        root = _lower_source(query.from_table)
        for index, join in enumerate(query.joins):
            root = Join(
                input=root,
                target=_lower_join_source(join),
                join_type=join.join_type,
                predicate=join.on,
            )
            if join.on is not None:
                root = _apply_expressions(root, ((join.on, f"join/{index}/on"),))

        if query.where is not None:
            root = _apply_expressions(root, ((query.where, "predicate/where"),))
            root = Selection(root, "WHERE", query.where)

        aggregates = _aggregate_expressions(query)
        if query.group_by or aggregates:
            root = _apply_expressions(
                root,
                tuple(
                    (expression, f"group_by/{index}")
                    for index, expression in enumerate(query.group_by)
                ),
            )
            root = GroupAggregate(root, query.group_by, aggregates)

        if query.having is not None:
            root = _apply_expressions(root, ((query.having, "predicate/having"),))
            root = Selection(root, "HAVING", query.having)

        windows = _window_expressions(query)
        if windows:
            root = Window(root, windows)

        for index, projection in enumerate(query.projections):
            root = _apply_expressions(
                root, ((projection.expr, f"projection/{index}"),)
            )
        root = Projection(root, query.projections)

        if query.distinct:
            root = DuplicateElimination(root)

        for index, item in enumerate(query.order_by):
            root = _apply_expressions(root, ((item.expr, f"order_by/{index}"),))
        if query.order_by:
            root = Sort(root, query.order_by)

        if query.limit is not None or query.offset is not None:
            root = Limit(root, query.limit, query.offset, query.limit_with_ties)

    return AlgebraQuery(
        root=root,
        ctes=tuple(lower_to_extended_algebra(cte) for cte in query.ctes),
        scope=query.scope,
        query_contract=query.query_contract,
    )


def decompose_algebra_query(query: AlgebraQuery) -> AlgebraQueryParts:
    """从输出节点向来源节点遍历代数树，提取因子化所需的节点载荷。"""

    if isinstance(query.root, SetOperation):
        return AlgebraQueryParts(
            source=None,
            joins=(),
            where=None,
            grouping_keys=(),
            aggregates=(),
            aggregate_references=(),
            having=None,
            windows=(),
            projections=(),
            distinct=False,
            order_by=(),
            limit=None,
            offset=None,
            limit_with_ties=False,
            set_operation=query.root,
            applications={},
        )

    where: BoundExpr | None = None
    grouping_keys: tuple[BoundExpr, ...] = ()
    aggregates: tuple[BoundFunc, ...] = ()
    having: BoundExpr | None = None
    windows: tuple[WindowExpression, ...] = ()
    projections: tuple[BoundProjection, ...] = ()
    distinct = False
    order_by: tuple[BoundOrderItem, ...] = ()
    limit: int | None = None
    offset: int | None = None
    limit_with_ties = False
    joins: list[Join] = []
    applications: dict[str, Apply] = {}
    node = query.root

    while True:
        if isinstance(node, Apply):
            applications[node.site] = node
            node = node.input
        elif isinstance(node, Limit):
            limit, offset, limit_with_ties = node.bound, node.offset, node.with_ties
            node = node.input
        elif isinstance(node, Sort):
            order_by = node.items
            node = node.input
        elif isinstance(node, DuplicateElimination):
            distinct = True
            node = node.input
        elif isinstance(node, Projection):
            projections = node.items
            node = node.input
        elif isinstance(node, Window):
            windows = node.expressions
            node = node.input
        elif isinstance(node, Selection):
            if node.clause == "WHERE":
                where = node.predicate
            else:
                having = node.predicate
            node = node.input
        elif isinstance(node, GroupAggregate):
            grouping_keys, aggregates = node.grouping_keys, node.aggregates
            node = node.input
        elif isinstance(node, Join):
            joins.append(node)
            node = node.input
        elif isinstance(node, (Relation, FromSubquery, NoRelation)):
            break
        else:
            raise TypeError(f"unsupported CADENCE algebra node: {type(node).__name__}")

    joins.reverse()
    aggregate_references = list(aggregates)
    for window in windows:
        aggregate_references.extend(_window_aggregate_expressions(window.expression))
    return AlgebraQueryParts(
        source=node,
        joins=tuple(joins),
        where=where,
        grouping_keys=grouping_keys,
        aggregates=aggregates,
        aggregate_references=tuple(aggregate_references),
        having=having,
        windows=windows,
        projections=projections,
        distinct=distinct,
        order_by=order_by,
        limit=limit,
        offset=offset,
        limit_with_ties=limit_with_ties,
        set_operation=None,
        applications=applications,
    )


def _lower_source(source: BoundTable | BoundDerivedTable | None) -> AlgebraNode:
    """把基础表、派生表或无 FROM 查询转换成来源节点。"""

    if source is None:
        return NoRelation()
    if isinstance(source, BoundTable):
        return Relation(source.relation_id, source.alias)
    return FromSubquery(
        lower_to_extended_algebra(source.query), source.alias, source.output_names
    )


def _lower_join_source(join: BoundJoin) -> Relation | FromSubquery:
    """把 JOIN 右侧转换为基础关系或 FROM 子查询节点。"""

    source = join.table
    if isinstance(source, BoundTable):
        return Relation(source.relation_id, source.alias)
    return FromSubquery(
        lower_to_extended_algebra(source.query), source.alias, source.output_names
    )


def _lower_set_operation(node: BoundSetOperation) -> SetOperation:
    """保留有序集合运算及其 ALL/DISTINCT 量词。"""

    return SetOperation(
        operator=node.operator,
        quantifier=node.quantifier,
        left=_lower_set_branch(node.left),
        right=_lower_set_branch(node.right),
    )


def _lower_set_branch(node: BoundQuery | BoundSetOperation) -> AlgebraQuery | SetOperation:
    """递归转换集合运算分支，并保留 SELECT 分支的独立作用域。"""

    if isinstance(node, BoundQuery):
        return lower_to_extended_algebra(node)
    return _lower_set_operation(node)


def _apply_expressions(
    input_node: AlgebraNode,
    expressions: tuple[tuple[BoundExpr, str], ...],
) -> AlgebraNode:
    """在所属关系节点上记录表达式内子查询的 Apply 位置。"""

    node = input_node
    for expression, path in expressions:
        for site, role, right, correlations in _apply_occurrences(expression, path):
            node = Apply(node, right, role, site, correlations)
    return node


def _apply_occurrences(
    expression: BoundExpr, path: str
) -> Iterator[tuple[str, str, AlgebraQuery | Relation, tuple[CorrelatedReference, ...]]]:
    """按绑定表达式顺序枚举嵌套查询与相关引用。"""

    if isinstance(expression, BoundIn):
        if expression.subquery is not None:
            query = lower_to_extended_algebra(expression.subquery.query)
            role = "NOT_IN_QUERY" if expression.negated else "IN_QUERY"
            yield path, role, query, query_correlations(expression.subquery.query)
            yield from _apply_occurrences(expression.operand, f"{path}/operand")
            return
        for index, value in enumerate(expression.values):
            yield from _apply_occurrences(value, f"{path}/value/{index}")
        yield from _apply_occurrences(expression.operand, f"{path}/operand")
        return
    if isinstance(expression, BoundScalarSubquery):
        query = lower_to_extended_algebra(expression.query)
        yield path, "SCALAR_QUERY", query, query_correlations(expression.query)
        return
    if isinstance(expression, BoundSubquery):
        query = lower_to_extended_algebra(expression.query)
        yield path, "SUBQUERY", query, query_correlations(expression.query)
        return
    if isinstance(expression, BoundNotExistsCorrelated):
        reference = CorrelatedReference(
            scope_depth=1,
            relation_id=expression.outer_column.relation_id,
            column_id=expression.outer_column.column_id,
            scope_path=None,
            projection_ordinal=None,
            inner_relation_id=expression.inner_column.relation_id,
            inner_column_id=expression.inner_column.column_id,
        )
        yield (
            path,
            "NOT_EXISTS_CORRELATED",
            Relation(expression.source_table.relation_id, expression.source_table.alias),
            (reference,),
        )
        return
    if isinstance(expression, BoundFunc):
        path_kind = "metric" if expression.name in CADENCE_AGGREGATE_FUNCTIONS else "arg"
        for index, argument in enumerate(expression.args):
            yield from _apply_occurrences(argument, f"{path}/{path_kind}/{index}")
        return
    if isinstance(expression, BoundWindow):
        yield from _apply_occurrences(expression.function, f"{path}/function")
        for index, item in enumerate(expression.partition_by):
            yield from _apply_occurrences(item, f"{path}/partition/{index}")
        for index, order_item in enumerate(expression.order_by):
            yield from _apply_occurrences(order_item.expr, f"{path}/order/{index}")
        yield from _frame_apply_occurrences(expression.frame, f"{path}/frame")
        return
    children: tuple[tuple[BoundExpr, str], ...]
    if isinstance(expression, BoundCast):
        children = ((expression.operand, f"{path}/operand"),)
    elif isinstance(expression, BoundBinary):
        children = (
            (expression.left, f"{path}/left"),
            (expression.right, f"{path}/right"),
        )
    elif isinstance(expression, BoundUnary):
        children = ((expression.operand, f"{path}/operand"),)
    elif isinstance(expression, BoundBetween):
        children = (
            (expression.operand, f"{path}/operand"),
            (expression.lower, f"{path}/lower"),
            (expression.upper, f"{path}/upper"),
        )
    elif isinstance(expression, BoundIsNull):
        children = ((expression.operand, f"{path}/operand"),)
    else:
        children = ()
    for child, child_path in children:
        yield from _apply_occurrences(child, child_path)


def _frame_apply_occurrences(
    frame: BoundWindowFrame | None, path: str
) -> Iterator[tuple[str, str, AlgebraQuery | Relation, tuple[CorrelatedReference, ...]]]:
    """枚举窗口 frame 偏移表达式中的子查询。"""

    if frame is None:
        return
    for name, bound in (("start", frame.start), ("end", frame.end)):
        yield from _bound_frame_apply_occurrences(bound, f"{path}/{name}")


def _bound_frame_apply_occurrences(
    bound: BoundWindowFrameBound, path: str
) -> Iterator[tuple[str, str, AlgebraQuery | Relation, tuple[CorrelatedReference, ...]]]:
    """枚举一个窗口边界偏移表达式中的子查询。"""

    if bound.offset is not None:
        yield from _apply_occurrences(bound.offset, f"{path}/offset")


def _aggregate_expressions(query: BoundQuery) -> tuple[BoundFunc, ...]:
    """按查询子句顺序收集分组聚合节点的聚合表达式。"""

    aggregates: list[BoundFunc] = []
    seen: set[BoundFunc] = set()
    for expression, _ in _query_expressions(query, include_join_predicates=True):
        for aggregate in _aggregates_in_expression(expression):
            if aggregate not in seen:
                aggregates.append(aggregate)
                seen.add(aggregate)
    return tuple(aggregates)


def _window_aggregate_expressions(window: BoundWindow) -> list[BoundFunc]:
    """收集窗口函数及其分区、排序和 frame 表达式中的聚合调用。"""

    expressions: list[BoundExpr] = [window.function, *window.partition_by]
    expressions.extend(item.expr for item in window.order_by)
    if window.frame is not None:
        for bound in (window.frame.start, window.frame.end):
            if bound.offset is not None:
                expressions.append(bound.offset)
    return [
        aggregate
        for expression in expressions
        for aggregate in _aggregates_in_expression(expression)
    ]


def _aggregates_in_expression(expression: BoundExpr) -> Iterator[BoundFunc]:
    """收集当前表达式作用域中的聚合，不穿透窗口或子查询作用域。"""

    if isinstance(expression, BoundWindow):
        return
    if isinstance(expression, BoundFunc):
        if expression.name in CADENCE_AGGREGATE_FUNCTIONS:
            yield expression
        for child in expression.args:
            yield from _aggregates_in_expression(child)
    elif isinstance(expression, BoundCast):
        yield from _aggregates_in_expression(expression.operand)
    elif isinstance(expression, BoundBinary):
        yield from _aggregates_in_expression(expression.left)
        yield from _aggregates_in_expression(expression.right)
    elif isinstance(expression, BoundUnary):
        yield from _aggregates_in_expression(expression.operand)
    elif isinstance(expression, BoundBetween):
        yield from _aggregates_in_expression(expression.operand)
        yield from _aggregates_in_expression(expression.lower)
        yield from _aggregates_in_expression(expression.upper)
    elif isinstance(expression, BoundIn):
        yield from _aggregates_in_expression(expression.operand)
        for child in expression.values:
            yield from _aggregates_in_expression(child)
    elif isinstance(expression, BoundIsNull):
        yield from _aggregates_in_expression(expression.operand)


def _window_expressions(query: BoundQuery) -> tuple[WindowExpression, ...]:
    """记录查询各子句中的窗口运算及其原始表达式路径。"""

    result: list[WindowExpression] = []
    for expression, path in _query_expressions(query, include_join_predicates=True):
        result.extend(_windows_in_expression(expression, path))
    return tuple(result)


def _windows_in_expression(expression: BoundExpr, path: str) -> list[WindowExpression]:
    """按因子化使用的路径递归收集窗口表达式。"""

    found: list[WindowExpression] = []
    if isinstance(expression, BoundWindow):
        found.append(WindowExpression(path, expression))
        found.extend(_windows_in_expression(expression.function, f"{path}/function"))
        for index, child in enumerate(expression.partition_by):
            found.extend(_windows_in_expression(child, f"{path}/partition/{index}"))
        for index, item in enumerate(expression.order_by):
            found.extend(_windows_in_expression(item.expr, f"{path}/order/{index}"))
        if expression.frame is not None:
            for name, bound in (("start", expression.frame.start), ("end", expression.frame.end)):
                if bound.offset is not None:
                    found.extend(
                        _windows_in_expression(bound.offset, f"{path}/frame/{name}/offset")
                    )
        return found
    for child, child_path in _expression_children(expression, path):
        found.extend(_windows_in_expression(child, child_path))
    return found


def _expression_children(expression: BoundExpr, path: str) -> tuple[tuple[BoundExpr, str], ...]:
    """按绑定 IR 的固定子节点次序返回标量表达式子项。"""

    if isinstance(expression, BoundFunc):
        path_kind = "metric" if expression.name in CADENCE_AGGREGATE_FUNCTIONS else "arg"
        return tuple(
            (child, f"{path}/{path_kind}/{index}")
            for index, child in enumerate(expression.args)
        )
    if isinstance(expression, BoundCast):
        return ((expression.operand, f"{path}/operand"),)
    if isinstance(expression, BoundBinary):
        return (
            (expression.left, f"{path}/left"),
            (expression.right, f"{path}/right"),
        )
    if isinstance(expression, BoundUnary):
        return ((expression.operand, f"{path}/operand"),)
    if isinstance(expression, BoundBetween):
        return (
            (expression.operand, f"{path}/operand"),
            (expression.lower, f"{path}/lower"),
            (expression.upper, f"{path}/upper"),
        )
    if isinstance(expression, BoundIn):
        children = tuple(
            (child, f"{path}/value/{index}")
            for index, child in enumerate(expression.values)
        )
        return (*children, (expression.operand, f"{path}/operand"))
    if isinstance(expression, BoundIsNull):
        return ((expression.operand, f"{path}/operand"),)
    return ()


def _query_expressions(
    query: BoundQuery, *, include_join_predicates: bool = False
) -> tuple[tuple[BoundExpr, str], ...]:
    """返回本查询作用域表达式及其稳定路径，不递归进入子查询。"""

    expressions: list[tuple[BoundExpr, str]] = []
    if include_join_predicates:
        for index, join in enumerate(query.joins):
            if join.on is not None:
                expressions.append((join.on, f"join/{index}/on"))
    expressions.extend(
        (projection.expr, f"projection/{index}")
        for index, projection in enumerate(query.projections)
    )
    expressions.extend(
        (expression, f"group_by/{index}")
        for index, expression in enumerate(query.group_by)
    )
    if query.where is not None:
        expressions.append((query.where, "predicate/where"))
    if query.having is not None:
        expressions.append((query.having, "predicate/having"))
    expressions.extend(
        (item.expr, f"order_by/{index}")
        for index, item in enumerate(query.order_by)
    )
    return tuple(expressions)


def query_correlations(query: BoundQuery) -> tuple[CorrelatedReference, ...]:
    """收集当前查询作用域中的外层列引用，不穿透其子查询。"""

    result: list[CorrelatedReference] = []
    for expression, _ in _query_expressions(query, include_join_predicates=True):
        result.extend(_correlations_in_expression(expression))
    return tuple(result)


def _correlations_in_expression(expression: BoundExpr) -> list[CorrelatedReference]:
    """提取标量表达式中的相关列；嵌套查询由各自 Apply 节点记录。"""

    if isinstance(expression, BoundOuterColumn):
        return [
            CorrelatedReference(
                scope_depth=expression.scope_depth,
                relation_id=expression.relation_id,
                column_id=expression.column_id,
                scope_path=expression.scope_path,
                projection_ordinal=expression.projection_ordinal,
            )
        ]
    if isinstance(expression, BoundNotExistsCorrelated):
        return [
            CorrelatedReference(
                scope_depth=1,
                relation_id=expression.outer_column.relation_id,
                column_id=expression.outer_column.column_id,
                scope_path=None,
                projection_ordinal=None,
                inner_relation_id=expression.inner_column.relation_id,
                inner_column_id=expression.inner_column.column_id,
            )
        ]
    if isinstance(expression, BoundIn):
        nested = [_correlations_in_expression(expression.operand)]
        nested.extend(_correlations_in_expression(item) for item in expression.values)
    elif isinstance(expression, BoundWindow):
        nested = [_correlations_in_expression(expression.function)]
        nested.extend(
            _correlations_in_expression(item) for item in expression.partition_by
        )
        nested.extend(
            _correlations_in_expression(item.expr) for item in expression.order_by
        )
        if expression.frame is not None:
            for bound in (expression.frame.start, expression.frame.end):
                if bound.offset is not None:
                    nested.append(_correlations_in_expression(bound.offset))
    else:
        nested = [
            _correlations_in_expression(child)
            for child, _ in _expression_children(expression, "")
        ]
    return [reference for values in nested for reference in values]
