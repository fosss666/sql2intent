"""Read-only SQL parsing and schema binding for candidate queries."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import sqlglot
import sqlglot.expressions as exp

from cadence.canonical_serialize import canonical_bytes
from cadence.sqlglot_ast_mapper import map_sqlite_group_concat_ast

# 非有限数值字面量文本（cast 到浮点类型时 fail-closed 为 BoundUnsupported）。
# 这些值无法产生确定性 canonical 数值字节，且语义上不是可排序的有限量。
_NON_FINITE_NUMERIC_TEXTS: frozenset[str] = frozenset({"NaN", "Infinity", "-Infinity"})

# DATE cast 只使用规范化后的 YYYY-MM-DD；不明确的斜线日期字符串一律
# BoundUnsupported，不在本层猜测区域格式。
# TIMESTAMP cast 的规范化形状：YYYY-MM-DD[ T]HH:MM[:SS[.fff]]。

SQL_PARSE_FAILED = "SQL_PARSE_FAILED"
SQL_MULTI_STATEMENT = "SQL_MULTI_STATEMENT"
SQL_NOT_READ_ONLY = "SQL_NOT_READ_ONLY"
SQL_UNSUPPORTED_STATEMENT = "SQL_UNSUPPORTED_STATEMENT"
UNRESOLVED_RELATION = "UNRESOLVED_RELATION"
UNRESOLVED_COLUMN = "UNRESOLVED_COLUMN"
AMBIGUOUS_COLUMN = "AMBIGUOUS_COLUMN"
CATALOG_INVALID = "CATALOG_INVALID"
CATALOG_SNAPSHOT_HASH_MISMATCH = "CATALOG_SNAPSHOT_HASH_MISMATCH"
CADENCE_PREDICATE_SOURCE_UNBOUND = "CADENCE_PREDICATE_SOURCE_UNBOUND"
CADENCE_PREDICATE_CORRELATED_SUBQUERY = "CADENCE_PREDICATE_CORRELATED_SUBQUERY"
CADENCE_PREDICATE_SUBQUERY_DEPTH_UNSUPPORTED = "CADENCE_PREDICATE_SUBQUERY_DEPTH_UNSUPPORTED"
CADENCE_PREDICATE_SUBQUERY_FILTER_UNSUPPORTED = "CADENCE_PREDICATE_SUBQUERY_FILTER_UNSUPPORTED"
CADENCE_PREDICATE_NOT_EXISTS_CORRELATION_UNBOUND = (
    "CADENCE_PREDICATE_NOT_EXISTS_CORRELATION_UNBOUND"
)
CADENCE_PREDICATE_NOT_EXISTS_SHAPE_UNSUPPORTED = "CADENCE_PREDICATE_NOT_EXISTS_SHAPE_UNSUPPORTED"
CADENCE_PREDICATE_ANTI_JOIN_SHAPE_UNPROVEN = "CADENCE_PREDICATE_ANTI_JOIN_SHAPE_UNPROVEN"
CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED = "CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED"
CADENCE_PREDICATE_SCALAR_SUBQUERY_CORRELATED = "CADENCE_PREDICATE_SCALAR_SUBQUERY_CORRELATED"
CADENCE_PREDICATE_QUERY_EXPRESSION_UNSUPPORTED = "CADENCE_PREDICATE_QUERY_EXPRESSION_UNSUPPORTED"

QUERY_CONTRACT_CADENCE = "CADENCE_QUERY_SHAPE"

_QUERY_TREE_FUNCTION_ARITY: dict[str, tuple[int, int]] = {
    "count": (1, 1), "sum": (1, 1), "avg": (1, 1), "min": (1, 1), "max": (1, 1),
    "julianday": (1, 1), "substring": (2, 3), "and": (2, 2), "or": (2, 2),
}
_QUERY_TREE_EXTENDED_FUNCTION_ARITY: dict[str, tuple[int, int]] = {
    **_QUERY_TREE_FUNCTION_ARITY,
    # 仅登记会出现在已绑定 SQLite AST 中的常见标量函数；未知函数仍 fail-closed。
    "coalesce": (2, 64), "replace": (3, 3), "str_position": (2, 2),
    "time_to_str": (2, 2), "ts_or_ds_to_timestamp": (1, 1),
    "group_concat": (1, 2),
}
_QUERY_TREE_BINARY_OPERATORS = frozenset({
    "eq", "neq", "lt", "lte", "gt", "gte", "add", "sub", "mul", "div", "mod",
})
_QUERY_TREE_EXTENDED_BINARY_OPERATORS = _QUERY_TREE_BINARY_OPERATORS | {
    "like", "ilike", "concat",
}
_QUERY_TREE_UNARY_OPERATORS = frozenset({"not", "negate"})


def _supports_query_shapes(query_contract: str) -> bool:
    return query_contract == QUERY_CONTRACT_CADENCE


def _supports_query_tree(query_contract: str) -> bool:
    return query_contract == QUERY_CONTRACT_CADENCE


def _supports_query_extensions(query_contract: str) -> bool:
    return query_contract == QUERY_CONTRACT_CADENCE


def _supports_expression_extensions(query_contract: str) -> bool:
    return query_contract == QUERY_CONTRACT_CADENCE


def _supports_query_sources(query_contract: str) -> bool:
    return query_contract == QUERY_CONTRACT_CADENCE


def _supports_ranking_windows(query_contract: str) -> bool:
    return query_contract == QUERY_CONTRACT_CADENCE


def _supports_correlated_scalar_queries(query_contract: str) -> bool:
    return query_contract == QUERY_CONTRACT_CADENCE


def _supports_sqlite_duplicate_aliases(
    query_contract: str, dialect: str
) -> bool:
    """按 SQLite 列唯一归属解析重复 FROM alias。"""
    return (
        query_contract == QUERY_CONTRACT_CADENCE
        and dialect.casefold() == "sqlite"
    )


def _supports_projection_stars(query_contract: str) -> bool:
    return query_contract == QUERY_CONTRACT_CADENCE


def _supports_group_concat(query_contract: str) -> bool:
    return query_contract == QUERY_CONTRACT_CADENCE

# 顶层写语句类型（只读守卫拒绝清单）。
_MUTATING_STATEMENTS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Merge,
    exp.Command,
    exp.Grant,
    exp.Revoke,
    exp.TruncateTable,
    exp.Comment,
)

# 只读但超出单 SELECT block 域的顶层类型。
_READ_ONLY_OUT_OF_DOMAIN = (exp.Union, exp.Intersect, exp.Except, exp.SetOperation)


class ParseBindError(Exception):
    """解析/绑定失败；``reason_code`` 恒为本模块封闭失败码。"""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(reason_code if not detail else f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


def _fail(reason_code: str, detail: str) -> ParseBindError:
    return ParseBindError(reason_code, detail)


# ---------------------------------------------------------------------------
# BoundQuery：确定性已绑定中间表示（消费方：任务 4 因子提取）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BoundExpr:
    """表达式基类；具体节点见子类。kind 用于无反射判别。"""

    kind: str


@dataclass(frozen=True)
class BoundColumn(BoundExpr):
    """已绑定列：relation_id + column_id（列引用语义）。"""

    relation_id: str
    column_id: str

    def __init__(self, relation_id: str, column_id: str) -> None:
        super().__init__("column")
        object.__setattr__(self, "relation_id", relation_id)
        object.__setattr__(self, "column_id", column_id)


@dataclass(frozen=True)
class BoundDerivedColumn(BoundExpr):
    """局部 derived table 的输出槽；用 scope path 与序号绑定，不借用显示 alias。"""

    scope_path: str
    projection_ordinal: int

    def __init__(self, scope_path: str, projection_ordinal: int) -> None:
        super().__init__("derived_column")
        object.__setattr__(self, "scope_path", scope_path)
        object.__setattr__(self, "projection_ordinal", projection_ordinal)


@dataclass(frozen=True)
class BoundOuterColumn(BoundExpr):
    """闭合子查询中的 lexical outer ref；identity 只含 scope depth 与绑定来源。"""

    scope_depth: int
    relation_id: str | None
    column_id: str | None
    scope_path: str | None
    projection_ordinal: int | None

    def __init__(
        self,
        scope_depth: int,
        *,
        relation_id: str | None = None,
        column_id: str | None = None,
        scope_path: str | None = None,
        projection_ordinal: int | None = None,
    ) -> None:
        super().__init__("outer_column")
        object.__setattr__(self, "scope_depth", scope_depth)
        object.__setattr__(self, "relation_id", relation_id)
        object.__setattr__(self, "column_id", column_id)
        object.__setattr__(self, "scope_path", scope_path)
        object.__setattr__(self, "projection_ordinal", projection_ordinal)


@dataclass(frozen=True)
class BoundLiteral(BoundExpr):
    """SQL 字面量的已解析表示；typed_tag 标记类型，canonical 保留解析文本。"""

    typed_tag: str
    canonical: str | None

    def __init__(self, typed_tag: str, canonical: str | None) -> None:
        super().__init__("literal")
        object.__setattr__(self, "typed_tag", typed_tag)
        object.__setattr__(self, "canonical", canonical)


@dataclass(frozen=True)
class BoundStar(BoundExpr):
    def __init__(self) -> None:
        super().__init__("star")


@dataclass(frozen=True)
class BoundFunc(BoundExpr):
    """函数调用；name 采用 sqlglot 规范化的函数类名（小写，方言间确定性），
    如 strftime → ``time_to_str``、sum → ``sum``。

    args 按 sqlglot 的 args 字典插入序收集（确定性），包含位置表达式参数与
    命名标量参数（如 time_to_str 的 format 字符串），保证身份字节完整；
    ``distinct`` 记录聚合 DISTINCT（sqlglot 以 Distinct 包装表达）。
    """

    name: str
    args: tuple[BoundExpr, ...]
    distinct: bool = False

    def __init__(self, name: str, args: tuple[BoundExpr, ...], distinct: bool = False) -> None:
        super().__init__("func")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "distinct", distinct)


@dataclass(frozen=True)
class BoundCast(BoundExpr):
    """目标类型显式的受限 CAST；不在绑定时转换值或推断列类型。"""

    target_type: str
    operand: BoundExpr

    def __init__(self, target_type: str, operand: BoundExpr) -> None:
        super().__init__("cast")
        object.__setattr__(self, "target_type", target_type)
        object.__setattr__(self, "operand", operand)


@dataclass(frozen=True)
class BoundWindowFrameBound:
    """CADENCE 查询范围内窗口 frame 的边界；offset 保留为绑定表达式。"""

    kind: Literal["UNBOUNDED", "CURRENT_ROW", "OFFSET"]
    side: Literal["PRECEDING", "FOLLOWING"] | None
    offset: BoundExpr | None


@dataclass(frozen=True)
class BoundWindowFrame:
    """解析器支持的 ROWS/RANGE frame 与 exclusion。"""

    unit: Literal["ROWS", "RANGE"]
    start: BoundWindowFrameBound
    end: BoundWindowFrameBound
    exclusion: Literal["NO_OTHERS", "CURRENT_ROW", "GROUP", "TIES"]


@dataclass(frozen=True)
class BoundWindow(BoundExpr):
    """窗口 IR；只保留当前查询范围支持的 typed frame。"""

    function: BoundFunc
    partition_by: tuple[BoundExpr, ...]
    order_by: tuple[BoundOrderItem, ...]
    frame: BoundWindowFrame | None

    def __init__(
        self,
        function: BoundFunc,
        partition_by: tuple[BoundExpr, ...],
        order_by: tuple[BoundOrderItem, ...],
        frame: BoundWindowFrame | None = None,
    ) -> None:
        super().__init__("window")
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "partition_by", partition_by)
        object.__setattr__(self, "order_by", order_by)
        object.__setattr__(self, "frame", frame)


@dataclass(frozen=True)
class BoundBinary(BoundExpr):
    """二元运算；op 为小写运算符名（and/or/eq/neq/lt/lte/gt/gte/...）。"""

    op: str
    left: BoundExpr
    right: BoundExpr

    def __init__(self, op: str, left: BoundExpr, right: BoundExpr) -> None:
        super().__init__("binary")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)


@dataclass(frozen=True)
class BoundUnary(BoundExpr):
    """一元运算；op 取 not / negate。"""

    op: str
    operand: BoundExpr

    def __init__(self, op: str, operand: BoundExpr) -> None:
        super().__init__("unary")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "operand", operand)


@dataclass(frozen=True)
class BoundBetween(BoundExpr):
    operand: BoundExpr
    lower: BoundExpr
    upper: BoundExpr
    negated: bool

    def __init__(
        self, operand: BoundExpr, lower: BoundExpr, upper: BoundExpr, negated: bool
    ) -> None:
        super().__init__("between")
        object.__setattr__(self, "operand", operand)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "negated", negated)


@dataclass(frozen=True)
class BoundIn(BoundExpr):
    """membership：字面量列表或独立局部子查询，二者不可混用。

    NULL 保持 SQL 三值语义；不得由空 values 猜测空子查询或推断 EXISTS 等价。
    保留原有三参数构造；显式 subquery 才表示局部 SELECT 来源。内层列的
    lineage 由子查询结构保存，不加入外层列收集；来源混用必须 fail-closed。
    """

    operand: BoundExpr
    values: tuple[BoundExpr, ...]
    negated: bool
    subquery: BoundSubquery | None = None

    def __init__(
        self,
        operand: BoundExpr,
        values: tuple[BoundExpr, ...],
        negated: bool,
        subquery: BoundSubquery | None = None,
    ) -> None:
        if subquery is not None and (values or not isinstance(subquery, BoundSubquery)):
            raise _fail(CADENCE_PREDICATE_SOURCE_UNBOUND, "membership 来源混用或未绑定")
        super().__init__("in")
        object.__setattr__(self, "operand", operand)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "negated", negated)
        object.__setattr__(self, "subquery", subquery)


@dataclass(frozen=True)
class BoundNotExistsCorrelated(BoundExpr):
    """闭合的 ``NOT EXISTS`` 单表等值 correlation。

    只保存 inner/outer 列身份和来源关系，不把它改写成 membership 或
    anti-join。相关键上的 NULL 仍按 SQL 三值比较处理，而 EXISTS 本身的
    返回域由 successor canonical 明确标记为二值布尔存在性。
    """

    source_table: BoundTable
    inner_column: BoundColumn
    outer_column: BoundColumn

    def __init__(
        self, source_table: BoundTable, inner_column: BoundColumn, outer_column: BoundColumn
    ) -> None:
        super().__init__("not_exists_correlated")
        object.__setattr__(self, "source_table", source_table)
        object.__setattr__(self, "inner_column", inner_column)
        object.__setattr__(self, "outer_column", outer_column)


@dataclass(frozen=True)
class BoundIsNull(BoundExpr):
    operand: BoundExpr
    negated: bool

    def __init__(self, operand: BoundExpr, negated: bool) -> None:
        super().__init__("is_null")
        object.__setattr__(self, "operand", operand)
        object.__setattr__(self, "negated", negated)


@dataclass(frozen=True)
class BoundUnsupported(BoundExpr):
    """当前绑定器无法表达的 SQL 结构；因子提取阶段据此标记 factor 状态。"""

    construct: str
    detail: str

    def __init__(self, construct: str, detail: str) -> None:
        super().__init__("unsupported")
        object.__setattr__(self, "construct", construct)
        object.__setattr__(self, "detail", detail)


@dataclass(frozen=True)
class BoundProjection:
    expr: BoundExpr
    alias: str | None


@dataclass(frozen=True)
class BoundTable:
    relation_id: str
    alias: str | None


@dataclass(frozen=True)
class BoundDerivedTable:
    """一层局部 derived source；外层列按 scope 固定的投影 ordinal 绑定。"""

    scope_path: str
    alias: str | None
    query: BoundQuery
    output_names: tuple[str | None, ...]


@dataclass(frozen=True)
class BoundCTEDefinition:
    """非递归 CTE 的已绑定查询与输出列名；CTE 标识符只用于绑定，不进入 identity。"""

    query: BoundQuery
    output_names: tuple[str | None, ...]


@dataclass(frozen=True)
class BoundJoin:
    table: BoundTable | BoundDerivedTable
    join_type: str  # INNER / LEFT
    on: BoundExpr | None


@dataclass(frozen=True)
class BoundOrderItem:
    """ORDER BY 单项：表达式 + 排序方向 + 空值位置。

    - ``direction`` 封闭为 ASC/DESC；sqlglot 在 postgres 方言下已把隐式方向
      规范化（``order by a`` → ASC），本层不猜测。
    - ``null_order`` 封闭为 FIRST/LAST（收窄 因子注册表的 DIALECT_DEFAULT）。
      sqlglot 已把方言默认空值序规范化：ASC 默认 NULLS LAST、DESC 默认 NULLS FIRST，
      因此 ``order by a desc`` 与 ``order by a desc nulls first`` 产生同一字段值，
      满足"显式默认与隐式默认须产生同一 identity"。
    - 本节点不携带 alias、原始 SQL 文本或 source span（identity 字节纯净性）。
    """

    expr: BoundExpr
    direction: Literal["ASC", "DESC"]
    null_order: Literal["FIRST", "LAST"]


@dataclass(frozen=True)
class BoundQuery:
    """单 SELECT block 的已绑定表示（含同域可展开的简单 CTE）。

    group_by/order_by 元素顺序保留语义顺序；DISTINCT 由 ``distinct`` 标记。
    """

    projections: tuple[BoundProjection, ...]
    from_table: BoundTable | BoundDerivedTable | None
    joins: tuple[BoundJoin, ...]
    where: BoundExpr | None
    group_by: tuple[BoundExpr, ...]
    order_by: tuple[BoundOrderItem, ...]
    ctes: tuple[BoundQuery, ...]
    distinct: bool
    # 没有这些子句时保持 None，使不含对应结构的 canonical identity 不变。
    having: BoundExpr | None = None
    compound: BoundSetOperation | None = None
    # query-tree identity 保留行数限制；未使用时既有 canonical 字节保持不变。
    limit: int | None = None
    offset: int | None = None
    query_contract: str = QUERY_CONTRACT_CADENCE
    scope: Literal["GLOBAL", "LOCAL"] = "GLOBAL"
    # 只供 CADENCE 查询范围使用；留在末尾以保持已有位置参数含义。
    limit_with_ties: bool = False


@dataclass(frozen=True)
class BoundSetOperation:
    """有序二元 set-operation node；operator、association 与 DISTINCT/ALL 均显式保留。"""

    operator: Literal["UNION", "INTERSECT", "EXCEPT"]
    quantifier: Literal["DISTINCT", "ALL"]
    left: BoundQuery | BoundSetOperation
    right: BoundQuery | BoundSetOperation


@dataclass(frozen=True)
class BoundSubquery(BoundExpr):
    """单列单表 SELECT 的局部作用域；不保存 SQL，不携带 outer scope。

    仅供 successor membership 使用；NULL/空集由 SQL 三值语义解释，禁止
    转成 EXISTS 或 anti-join。canonical 再检查域，防止手工 IR 绕过 binder。
    query 保存内层表/列身份，作为局部 lineage 来源；不继承外层 alias 或 CTE，
    不物化集合，不为回答阶段补候选或重新生成 SQL 提供入口。
    """

    query: BoundQuery

    def __init__(self, query: BoundQuery) -> None:
        super().__init__("subquery")
        object.__setattr__(self, "query", query)


@dataclass(frozen=True)
class BoundScalarSubquery(BoundExpr):
    """闭合标量查询 IR；AVG 与 COUNT 的空输入语义由 typed projection 区分。"""

    query: BoundQuery

    def __init__(self, query: BoundQuery) -> None:
        super().__init__("scalar_subquery")
        object.__setattr__(self, "query", query)


def canonical_bound_subquery(subquery: BoundSubquery) -> dict[str, Any]:
    """输出封闭局部 SELECT 身份；只含绑定列/表，不含别名或原 SQL。

    只允许同一投影列上的 ``IS NOT NULL`` 过滤；其他过滤、聚合、多列、嵌套
    及关联均拒绝。这样既能保留当前 NOT IN 候选的显式 NULL 过滤，也不会
    把任意 WHERE 文本压扁成未知集合。
    保留集合来源身份而不物化结果，NULL 不被删除，也不跨 negative forms 归一化。
    """
    query = subquery.query
    if (
        not isinstance(query.from_table, BoundTable)
        or len(query.projections) != 1 or query.group_by
        or query.order_by or query.ctes or query.distinct is not False
        or query.having is not None or query.compound is not None
        or not isinstance(query.projections[0].expr, BoundColumn)
    ):
        raise ValueError(CADENCE_PREDICATE_SOURCE_UNBOUND)
    column = query.projections[0].expr
    result: dict[str, Any] = {
        "kind": "SELECT", "scope": "LOCAL",
        "column": {"relation_id": column.relation_id, "column_id": column.column_id},
        "from_relation": {"relation_id": query.from_table.relation_id},
    }
    if query.joins:
        if len(query.joins) != 1:
            raise ValueError(CADENCE_PREDICATE_SOURCE_UNBOUND)
        join = query.joins[0]
        if (
            join.join_type != "INNER" or not isinstance(join.on, BoundBinary)
            or not isinstance(join.table, BoundTable)
            or join.on.op != "eq" or not isinstance(join.on.left, BoundColumn)
            or not isinstance(join.on.right, BoundColumn)
            or join.on.left.relation_id == join.on.right.relation_id
            or column.relation_id != query.from_table.relation_id
        ):
            raise ValueError(CADENCE_PREDICATE_SOURCE_UNBOUND)
        result["joins"] = [{
            "relation": {"relation_id": join.table.relation_id},
            "join_type": join.join_type,
            "on": _bound_to_canonical(join.on),
        }]
        if query.where is not None:
            result["where"] = _bound_to_canonical(query.where)
        return result
    if column.relation_id != query.from_table.relation_id:
        raise ValueError(CADENCE_PREDICATE_CORRELATED_SUBQUERY)
    if query.where is not None:
        if not (
            isinstance(query.where, BoundIsNull)
            and query.where.negated
            and isinstance(query.where.operand, BoundColumn)
            and query.where.operand == column
        ):
            raise ValueError(CADENCE_PREDICATE_SUBQUERY_FILTER_UNSUPPORTED)
        result["filter"] = {
            "tag": "IS_NULL",
            "value": {
                "tag": "BOUND_COLUMN",
                "value": {"relation_id": column.relation_id, "column_id": column.column_id},
            },
            "negated": True,
        }
    return result


# ---------------------------------------------------------------------------
# SchemaCatalog：数据库快照只读目录
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CatalogRelation:
    relation_id: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class SchemaCatalog:
    """只读关系目录；构造时校验唯一性，查询时大小写折叠匹配（确定性）。

    ``snapshot_hash`` 以 规范序列化 规范字节计算（见
    ``_catalog_canonical``），用于 B4 数据库快照哈希一致性校验。
    """

    dialect_id: str
    relations: Mapping[str, CatalogRelation]
    _by_lower: Mapping[str, str]
    _cols_by_lower: Mapping[str, Mapping[str, str]]

    @classmethod
    def from_definition(cls, definition: Mapping[str, Any]) -> SchemaCatalog:
        """从 catalog 定义构造；定义形如
        ``{"dialect_id": ..., "relations": [{"relation_id": ..., "columns": [...]}]}``。"""
        dialect_id = definition.get("dialect_id")
        if not isinstance(dialect_id, str) or not dialect_id:
            raise _fail(CATALOG_INVALID, "dialect_id 必须为非空字符串")
        rels_raw = definition.get("relations")
        if not isinstance(rels_raw, list):
            raise _fail(CATALOG_INVALID, "relations 必须为数组")
        relations: dict[str, CatalogRelation] = {}
        by_lower: dict[str, str] = {}
        for item in rels_raw:
            rid = item.get("relation_id")
            cols = item.get("columns")
            if not isinstance(rid, str) or not rid:
                raise _fail(CATALOG_INVALID, "relation_id 必须为非空字符串")
            if not isinstance(cols, list) or not cols or not all(
                isinstance(c, str) and c for c in cols
            ):
                raise _fail(CATALOG_INVALID, f"{rid} 的 columns 必须为非空字符串数组")
            key = rid.lower()
            if key in by_lower:
                raise _fail(CATALOG_INVALID, f"重复 relation_id（大小写折叠后）: {rid}")
            seen: set[str] = set()
            for c in cols:
                if c.lower() in seen:
                    raise _fail(CATALOG_INVALID, f"{rid} 重复 column_id: {c}")
                seen.add(c.lower())
            by_lower[key] = rid
            relations[rid] = CatalogRelation(relation_id=rid, columns=tuple(cols))
        cols_by_lower: dict[str, Mapping[str, str]] = {}
        for rid, rel in relations.items():
            cols_by_lower[rid] = {c.lower(): c for c in rel.columns}
        return cls(
            dialect_id=dialect_id,
            relations=relations,
            _by_lower=by_lower,
            _cols_by_lower=cols_by_lower,
        )

    def relation_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.relations))

    def resolve_relation(self, name: str) -> str | None:
        """SQL 表名 → 规范 relation_id；未命中返回 None。"""
        return self._by_lower.get(name.lower())

    def resolve_column(self, relation_id: str, name: str) -> str | None:
        """(relation_id, SQL 列名) → 规范 column_id；未命中返回 None。"""
        table = self._cols_by_lower.get(relation_id)
        if table is None:
            return None
        return table.get(name.lower())

    def snapshot_hash(self) -> str:
        """catalog 的 规范序列化 规范字节 SHA-256。"""
        return hashlib.sha256(canonical_bytes(_catalog_canonical(self))).hexdigest()


def _catalog_canonical(catalog: SchemaCatalog) -> dict[str, Any]:
    """catalog 快照的唯一规范形态（键排序交给 canonical_bytes）。

    只包含决定绑定结果的稳定身份：dialect_id 与 relation/column 身份，
    不包含显示名、别名或运行时元数据。
    """
    return {
        "dialect_id": catalog.dialect_id,
        "relations": [
            {"relation_id": rid, "columns": list(catalog.relations[rid].columns)}
            for rid in sorted(catalog.relations)
        ],
    }


# ---------------------------------------------------------------------------
# ReadOnlyGuard
# ---------------------------------------------------------------------------
class ReadOnlyGuard:
    """只读守卫：接受单 read-only SELECT block（含简单 CTE），其余 fail-closed。"""

    def __init__(self, dialect: str = "postgres") -> None:
        self._dialect = dialect

    def check(
        self, sql: str, *, query_contract: str = QUERY_CONTRACT_CADENCE
    ) -> exp.Expression:
        try:
            statements = sqlglot.parse(sql, read=self._dialect)
        except Exception as exc:  # noqa: BLE001 - 统一收敛为封闭失败码
            raise _fail(SQL_PARSE_FAILED, f"解析失败: {exc}") from exc
        if not statements:
            raise _fail(SQL_PARSE_FAILED, "空输入或无可解析语句")
        if len(statements) > 1:
            raise _fail(SQL_MULTI_STATEMENT, f"单条输入含 {len(statements)} 个语句")
        top = statements[0]
        if isinstance(top, _MUTATING_STATEMENTS):
            raise _fail(SQL_NOT_READ_ONLY, f"非只读语句: {type(top).__name__}")
        # sqlglot 也会把裸比较/函数等表达式解析成 AST；SQLite 还会把顶层括号
        # SELECT 解析为 Subquery，但 SQLite 不接受这两类形态作为完整 SQL 查询。
        # 把它们误报为“有效但超出身份域的 statement”会让一个无效生成候选触发
        # EvaluationPool 整池 fail-closed。VALUES/PRAGMA 等有效但未建模形态仍保留
        # SQL_UNSUPPORTED_STATEMENT，避免静默丢掉可能有效的语义候选。
        if isinstance(top, exp.Condition) or (
            isinstance(top, exp.Subquery) and self._dialect.lower() == "sqlite"
        ):
            raise _fail(SQL_PARSE_FAILED, "顶层表达式或括号查询不是完整 SQL 查询")
        if isinstance(top, _READ_ONLY_OUT_OF_DOMAIN) and not (
            (isinstance(top, exp.Union) and _supports_query_shapes(query_contract))
            or (_supports_query_tree(query_contract))
        ):
            raise _fail(
                SQL_UNSUPPORTED_STATEMENT,
                f"集合运算超出首版单 SELECT block 域: {type(top).__name__}",
            )
        if not isinstance(top, exp.Select) and not (
            (isinstance(top, exp.Union) and _supports_query_shapes(query_contract))
            or (_supports_query_tree(query_contract) and isinstance(top, exp.SetOperation))
        ):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, f"不支持的顶层语句: {type(top).__name__}")
        if top.find(exp.Into) is not None:
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "SELECT INTO 超出首版绑定域")
        return top


# ---------------------------------------------------------------------------
# Binder：标识符限定（未解析/歧义 fail-closed）
# ---------------------------------------------------------------------------
class _Binder:
    def __init__(
        self,
        catalog: SchemaCatalog,
        *,
        query_contract: str = QUERY_CONTRACT_CADENCE,
        dialect: str = "postgres",
        scope_path: str = "root",
        scope: Literal["GLOBAL", "LOCAL"] = "GLOBAL",
        query_depth: int = 0,
        cte_definitions: Mapping[str, BoundCTEDefinition] | None = None,
        outer_scopes: tuple[
            Mapping[str, tuple[str | BoundDerivedTable, ...]], ...
        ] = (),
    ) -> None:
        self._catalog = catalog
        self._query_contract = query_contract
        self._dialect = dialect.lower()
        self._scope_path = scope_path
        self._scope = scope
        self._query_depth = query_depth
        self._scope_serial = 0
        # 当前 FROM 作用域：alias(lower) -> relation_id。
        self._tables: dict[str, str] = {}
        # 重复 SQLite alias 在显式 successor 中保留所有来源，以列名唯一归属绑定。
        self._alias_sources: dict[str, list[str]] = {}
        # SELECT * 必须按 SQL FROM/JOIN 次序展开；alias->source 映射会折叠重复别名。
        self._source_order: list[tuple[str, str]] = []
        # derived scope path -> typed inner query; display aliases never enter identity.
        self._derived_scopes: dict[str, BoundDerivedTable] = {}
        # 仅用于当前 SQL scope 的 CTE name lookup；身份通过内联 query tree 表达。
        self._cte_definitions = dict(cte_definitions or {})
        self._cte_inline: dict[str, str] = {}
        self._outer_scopes = outer_scopes

    def _require_query_tree_depth(self) -> None:
        """限制局部查询嵌套深度，避免从样本语法扩成无界解析器。"""
        max_depth = 3 if _supports_query_sources(self._query_contract) else 1
        if self._query_depth >= max_depth:
            raise _fail(
                CADENCE_PREDICATE_SUBQUERY_DEPTH_UNSUPPORTED,
                f"当前 query contract 最多允许 {max_depth} 层局部 query",
            )

    def bind(self, node: exp.Expression) -> BoundQuery:
        node = self._bind_and_strip_ctes(node)
        if isinstance(node, exp.SetOperation):
            if not _supports_query_shapes(self._query_contract):
                raise _fail(
                    SQL_UNSUPPORTED_STATEMENT,
                    "集合运算仅属于当前 CADENCE 查询范围",
                )
            if (
                not isinstance(node, exp.Union)
                and not _supports_query_tree(self._query_contract)
            ):
                raise _fail(
                    SQL_UNSUPPORTED_STATEMENT,
                    "INTERSECT/EXCEPT 仅属于 当前 CADENCE 查询范围",
                )
            compound = self._bind_set_tree(node)
            return BoundQuery(
                projections=(), from_table=None, joins=(), where=None, group_by=(),
                order_by=(), ctes=(), distinct=False, compound=compound,
                query_contract=self._query_contract,
                scope=self._scope,
            )
        if not isinstance(node, exp.Select):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, f"不支持 SELECT leaf: {type(node).__name__}")
        return self._bind_select(node)

    def _bind_and_strip_ctes(self, node: exp.Expression) -> exp.Expression:
        """绑定 WITH scope，并把引用展开为 query source。

        展开只发生在绑定 IR，不改写候选 SQL；递归、物化提示和带搜索子句的
        CTE 不受支持，避免把不同执行语义压成同一 identity。
        """
        with_node = node.args.get("with_")
        if with_node is None or not _supports_query_sources(self._query_contract):
            return node
        if not isinstance(with_node, exp.With):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "WITH 节点类型超域")
        if with_node.args.get("recursive"):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "不支持递归 CTE")
        if with_node.args.get("search") is not None:
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "不支持 CTE SEARCH 子句")

        definitions = dict(self._cte_definitions)
        for ordinal, cte in enumerate(with_node.args.get("expressions") or []):
            if not isinstance(cte, exp.CTE):
                raise _fail(SQL_UNSUPPORTED_STATEMENT, "WITH 项必须是已声明 CTE")
            for key, value in cte.args.items():
                if key not in {"this", "alias", "materialized", "key_expressions"}:
                    if value is not None:
                        raise _fail(SQL_UNSUPPORTED_STATEMENT, f"CTE 修饰超域: {key}")
                elif key == "materialized" and value is not None:
                    # sqlglot 用 False 表示显式 NOT MATERIALIZED；把它当默认值会
                    # 把执行提示抹掉，导致不同 CTE 语义进入同一 identity。
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, f"CTE 修饰超域: {key}")
                elif key == "key_expressions" and value not in (None, False, []):
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, f"CTE 修饰超域: {key}")
            cte_name = cte.alias
            if not cte_name:
                raise _fail(SQL_UNSUPPORTED_STATEMENT, "CTE 名称不能为空")
            key = cte_name.casefold()
            if key in definitions:
                raise _fail(SQL_UNSUPPORTED_STATEMENT, f"CTE 名称重复或遮蔽: {cte_name!r}")
            inner = cte.this
            if isinstance(inner, exp.Subquery):
                if any(value for arg, value in inner.args.items() if arg != "this"):
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, "CTE query wrapper 修饰超域")
                inner = inner.this
            if not isinstance(inner, (exp.Select, exp.SetOperation)):
                raise _fail(SQL_UNSUPPORTED_STATEMENT, "CTE 定义必须是 SELECT/query tree")
            self._require_query_tree_depth()
            local = _Binder(
                self._catalog,
                query_contract=self._query_contract,
                dialect=self._dialect,
                scope_path=f"{self._scope_path}/cte/{ordinal}",
                scope="LOCAL",
                query_depth=self._query_depth + 1,
                cte_definitions=definitions,
            )
            query = local.bind(inner)
            projected_names = _query_projection_names(query)
            alias_node = cte.args.get("alias")
            alias_columns = (
                alias_node.args.get("columns") or []
                if isinstance(alias_node, exp.TableAlias)
                else []
            )
            if alias_columns:
                output_names = tuple(
                    column.name if isinstance(column, exp.Identifier) else None
                    for column in alias_columns
                )
                if len(output_names) != _query_projection_count(query):
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, "CTE 列名数量与投影列数不一致")
            else:
                output_names = projected_names
            if len(output_names) != _query_projection_count(query):
                raise _fail(SQL_UNSUPPORTED_STATEMENT, "CTE 输出列数无法唯一绑定")
            if any(name is None or not name for name in output_names):
                raise _fail(SQL_UNSUPPORTED_STATEMENT, "CTE 输出列必须具有可绑定名称")
            definitions[key] = BoundCTEDefinition(query, output_names)

        self._cte_definitions = definitions
        stripped = node.copy()
        stripped.set("with_", None)
        return stripped

    def _bind_select(self, node: exp.Select) -> BoundQuery:
        if _supports_query_tree(self._query_contract):
            allowed_args = {
                "kind", "hint", "distinct", "expressions", "limit", "offset",
                "operation_modifiers", "from_", "where", "group", "having",
                "order", "joins", "with_",
            }
            for key, value in node.args.items():
                if key not in allowed_args and value is not None:
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, f"SELECT 包含不支持的修饰: {key}")
            if node.args.get("with_") is not None:
                raise _fail(SQL_UNSUPPORTED_STATEMENT, "当前 query tree 不接受 CTE")
        if node.args.get("having") is not None and not _supports_query_shapes(
            self._query_contract
        ):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "当前查询形态不支持 HAVING")
        cte_items = [self._bind_cte(c) for c in self._ctes(node)]
        ctes = tuple(query for _, query in cte_items)
        self._cte_inline = self._inline_map(cte_items)
        distinct = node.args.get("distinct") is not None
        row_limit, row_offset, limit_with_ties = (
            self._row_limit(node)
            if _supports_query_tree(self._query_contract)
            else (None, None, False)
        )
        from_node = node.args.get("from_")
        if from_node is None:
            return BoundQuery(
                projections=self._projections(node),
                from_table=None,
                joins=(),
                where=None,
                group_by=(),
                order_by=(),
                ctes=ctes,
                distinct=distinct,
                having=self._having(node),
                limit=row_limit,
                offset=row_offset,
                limit_with_ties=limit_with_ties,
                query_contract=self._query_contract,
                scope=self._scope,
            )
        self._tables = self._from_scope(from_node)
        from_table = self._from_table(from_node)
        joins = self._joins(node)
        where = self._where(node)
        group_by = self._group_by(node)
        projections = self._projections(node)
        order_by = self._order_by(node, projections=projections)
        return BoundQuery(
            projections=projections,
            from_table=from_table,
            joins=joins,
            where=where,
            group_by=group_by,
            order_by=order_by,
            ctes=ctes,
            distinct=distinct,
            having=self._having(node),
            limit=row_limit,
            offset=row_offset,
            limit_with_ties=limit_with_ties,
            query_contract=self._query_contract,
            scope=self._scope,
        )

    def _bind_set_tree(self, node: exp.Expression) -> BoundSetOperation:
        """绑定有序二元 set tree；operator/quantifier 与 AST 结合形状均保留。"""
        if not isinstance(node, exp.SetOperation):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "set-operation child 必须是已声明二元节点")
        operator = type(node).__name__.upper()
        if operator not in {"UNION", "INTERSECT", "EXCEPT"}:
            raise _fail(SQL_UNSUPPORTED_STATEMENT, f"未知 set operator: {operator}")
        if operator != "UNION" and not _supports_query_tree(self._query_contract):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, f"{operator} 仅属于 当前 CADENCE 查询范围")
        for key, value in node.args.items():
            if key not in {"this", "expression", "distinct", "by_name", "side", "kind", "on"}:
                if value is not None:
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, f"set-operation 修饰超域: {key}")
            elif key in {"by_name", "side", "kind", "on"} and value not in (None, False, ""):
                raise _fail(SQL_UNSUPPORTED_STATEMENT, f"set-operation 修饰超域: {key}")
        left_node, right_node = node.this, node.expression

        def bind_branch(branch: exp.Expression) -> BoundQuery | BoundSetOperation:
            if isinstance(branch, exp.Subquery):
                if any(value for key, value in branch.args.items() if key not in {"this", "alias"}):
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, "set-operation branch wrapper 超域")
                branch = branch.this
            if isinstance(branch, exp.SetOperation):
                return self._bind_set_tree(branch)
            if not isinstance(branch, exp.Select):
                raise _fail(
                    SQL_UNSUPPORTED_STATEMENT,
                    "set-operation branch 必须是 SELECT 或 set tree",
                )
            if not _supports_query_tree(self._query_contract):
                if branch.args.get("order") is not None or branch.args.get("limit") is not None:
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, "UNION 分支 ORDER BY/LIMIT 不受支持")
                if branch.args.get("distinct") is not None:
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, "UNION 分支 DISTINCT 修饰不受支持")
            return self._bind_select(branch)

        left, right = bind_branch(left_node), bind_branch(right_node)

        def projection_count(branch: BoundQuery | BoundSetOperation) -> int:
            return (
                len(branch.projections)
                if isinstance(branch, BoundQuery)
                else projection_count(branch.left)
            )

        if projection_count(left) != projection_count(right):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "set-operation branch 投影列数不一致")
        return BoundSetOperation(
            operator=operator,  # type: ignore[arg-type]  # sqlglot type is runtime-dispatched
            quantifier="DISTINCT" if node.args.get("distinct") is not False else "ALL",
            left=left,
            right=right,
        )

    @staticmethod
    def _ctes(node: exp.Select) -> list[exp.CTE]:
        with_node = node.args.get("with_")
        if with_node is None:
            return []
        if with_node.args.get("recursive"):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "递归 CTE 超出首版绑定域")
        expressions = with_node.args.get("expressions") or []
        return [c for c in expressions if isinstance(c, exp.CTE)]

    def _bind_cte(self, cte: exp.CTE) -> tuple[str, BoundQuery]:
        """绑定单个 CTE 定义，返回 (CTE 名, BoundQuery)。

        CTE 名（``cte.alias``）必须随绑定结果一起返回，供外层 FROM
        引用解析（``_inline_map`` 以 CTE 名为键）。
        """
        inner = cte.this
        if not isinstance(inner, exp.Select):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "CTE 内非 SELECT，超出首版域")
        return cte.alias, self.bind(inner)

    @staticmethod
    def _inline_map(cte_items: list[tuple[str, BoundQuery]]) -> dict[str, str]:
        """简单 CTE 名 → 底层 relation_id；非简单 CTE（聚合投影/多源/复杂）拒绝。

        同域可展开 = 单 FROM 表、无 join/where/group/order/distinct/嵌套 CTE、
        投影全部为直接列引用。CTE 名大小写折叠为键。
        """
        out: dict[str, str] = {}
        for cte_name, cte in cte_items:
            ok = (
                cte.from_table is not None
                and not cte.joins
                and cte.where is None
                and not cte.group_by
                and not cte.order_by
                and not cte.distinct
                and not cte.ctes
                and cte.having is None
                and cte.compound is None
                and all(isinstance(p.expr, BoundColumn) for p in cte.projections)
            )
            if not ok:
                raise _fail(
                    SQL_UNSUPPORTED_STATEMENT,
                    "CTE 投影非直接列或多源，超出首版同域可展开域",
                )
            # ok 条件保证 from_table 非 None（上面已断言单 FROM 表）。
            assert cte.from_table is not None
            if not isinstance(cte.from_table, BoundTable):
                raise _fail(SQL_UNSUPPORTED_STATEMENT, "CTE 不能基于 derived source")
            out[cte_name.lower()] = cte.from_table.relation_id
        return out

    def _from_scope(self, from_node: exp.From) -> dict[str, str]:
        self._alias_sources = {}
        self._source_order = []
        scope: dict[str, str] = {}
        this = from_node.this
        if isinstance(this, exp.Subquery):
            if not _supports_query_tree(self._query_contract):
                raise _fail(SQL_UNSUPPORTED_STATEMENT, "FROM 子查询超出首版绑定域")
            self._register_derived(scope, this, scope_label="from")
        if isinstance(this, exp.Table):
            if (
                _supports_query_sources(self._query_contract)
                and this.name.casefold() in self._cte_definitions
            ):
                self._register_cte_source(scope, this, scope_label="from")
            else:
                self._register_table(scope, this)
        elif not isinstance(this, exp.Subquery):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, f"不支持 FROM 节点: {type(this).__name__}")
        return scope

    def _register_derived(
        self,
        scope: dict[str, str],
        node: exp.Subquery,
        *,
        scope_label: str = "from",
    ) -> None:
        """绑定一层非相关 derived query，并把外层列名映射到固定输出槽。"""
        if any(value for key, value in node.args.items() if key not in {"this", "alias"}):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "derived table 包装修饰超域")
        self._require_query_tree_depth()
        alias_node = node.args.get("alias")
        alias = node.alias or None
        if alias_node is not None and alias_node.args.get("columns"):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "derived table 列名列表超域")
        if not alias and self._dialect != "sqlite":
            raise _fail(
                SQL_UNSUPPORTED_STATEMENT,
                "当前 SQL dialect 要求 derived table alias",
            )
        key = alias.casefold() if alias else None
        query_node = node.this
        if not isinstance(query_node, (exp.Select, exp.SetOperation)):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "derived source 必须是 SELECT/query tree")
        scope_path = self._new_derived_scope_path(scope_label)
        local = _Binder(
            self._catalog,
            query_contract=self._query_contract,
            dialect=self._dialect,
            scope_path=scope_path,
            scope="LOCAL",
            query_depth=self._query_depth + 1,
            cte_definitions=self._cte_definitions,
        )
        try:
            query = local.bind(query_node)
        except ParseBindError as exc:
            if self._query_references_outer(query_node, local):
                raise _fail(
                    CADENCE_PREDICATE_CORRELATED_SUBQUERY,
                    "derived query 引用外层列",
                ) from exc
            raise
        if (
            _query_contains_derived_table(query)
            and not _supports_query_sources(self._query_contract)
        ):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "derived table 最多一层")
        count = _query_projection_count(query)
        names: list[str | None] = []
        leaves = _query_projection_names(query)
        if len(leaves) != count:
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "derived set-operation 输出列数不一致")
        names.extend(leaves)
        if any(name is None for name in names):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "derived output 必须有可绑定列名")
        derived = BoundDerivedTable(
            scope_path=scope_path,
            alias=alias,
            query=query,
            output_names=tuple(names),
        )
        self._derived_scopes[scope_path] = derived
        # SQLite 允许省略 derived-table alias；内部 key 只供本层 unqualified
        # 输出列解析使用，不序列化到 query tree，避免把实现细节混入 identity。
        self._record_scope_alias(
            scope,
            key if key is not None else f"\x00derived:{scope_path}",
            scope_path,
            alias or scope_path,
        )

    def _new_derived_scope_path(self, scope_label: str) -> str:
        """以 SQL source 位置生成稳定局部 scope，避免别名参与候选 identity。"""
        path = f"{self._scope_path}/{scope_label}"
        if path in self._derived_scopes:
            self._scope_serial += 1
            path = f"{path}/{self._scope_serial}"
        return path

    def _register_cte_source(
        self,
        scope: dict[str, str],
        table: exp.Table,
        *,
        scope_label: str,
    ) -> None:
        """把非递归 CTE 引用绑定为局部 query source，不把 CTE 名写入 identity。"""
        if table.args.get("alias") is not None and table.args["alias"].args.get("columns"):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "CTE 引用别名列列表超域")
        alias = table.alias or table.name
        key = alias.casefold()
        definition = self._cte_definitions[table.name.casefold()]
        scope_path = self._new_derived_scope_path(scope_label)
        derived = BoundDerivedTable(
            scope_path=scope_path,
            alias=alias,
            query=definition.query,
            output_names=definition.output_names,
        )
        self._derived_scopes[scope_path] = derived
        self._record_scope_alias(scope, key, scope_path, alias)

    def _query_references_outer(self, node: exp.Expression, local: _Binder) -> bool:
        """只检查当前 query level，区分显式 outer reference 与普通未解析列。"""
        for column in _columns_at_current_query_level(node):
            try:
                local._column(column)
            except ParseBindError:
                try:
                    self._column(column)
                except ParseBindError:
                    continue
                return True
        return False

    def _resolve_relation_id(self, name: str) -> str | None:
        rid = self._catalog.resolve_relation(name)
        if rid is not None:
            return rid
        return self._cte_inline.get(name.lower())

    def _register_table(self, scope: dict[str, str], table: exp.Table) -> None:
        rid = self._resolve_relation_id(table.name)
        if rid is None:
            raise _fail(UNRESOLVED_RELATION, f"未解析关系: {table.name!r}")
        alias = table.alias or table.name
        key = alias.casefold()
        self._record_scope_alias(scope, key, rid, alias)

    def _record_scope_alias(
        self,
        scope: dict[str, str],
        key: str,
        source_id: str,
        alias_label: str,
    ) -> None:
        """记录当前 SQL scope 的来源；仅 SQLite successor 保留同名 alias 多来源。"""
        existing = self._alias_sources.get(key)
        if existing is None:
            existing = [scope[key]] if key in scope else []
            self._alias_sources[key] = existing
        if existing and not _supports_sqlite_duplicate_aliases(
            self._query_contract, self._dialect
        ):
            raise _fail(AMBIGUOUS_COLUMN, f"FROM 作用域重复别名: {alias_label!r}")
        existing.append(source_id)
        scope[key] = source_id
        self._source_order.append((key, source_id))

    def _from_table(self, from_node: exp.From) -> BoundTable | BoundDerivedTable:
        table = from_node.this
        if isinstance(table, exp.Subquery):
            alias = table.alias
            if alias:
                scope_path = self._tables[alias.lower()]
            else:
                derived_paths = [
                    value for value in self._tables.values()
                    if value in self._derived_scopes
                ]
                if len(derived_paths) != 1:
                    raise _fail(
                        SQL_UNSUPPORTED_STATEMENT,
                        "derived source 无法唯一绑定局部 scope",
                    )
                scope_path = derived_paths[0]
            return self._derived_scopes[scope_path]
        assert isinstance(table, exp.Table)
        table_name = table.name
        if not isinstance(table_name, str) or not table_name:
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "CTE source 名称不能为空")
        if table_name.casefold() in self._cte_definitions:
            alias = table.alias or table_name
            resolved_scope_path = self._tables.get(alias.casefold())
            derived = self._derived_scopes.get(resolved_scope_path or "")
            if derived is None:
                raise _fail(SQL_UNSUPPORTED_STATEMENT, "CTE source 无法回指已绑定 scope")
            return derived
        rid = self._resolve_relation_id(table.name)
        assert rid is not None
        return BoundTable(relation_id=rid, alias=table.alias)

    def _joins(self, node: exp.Select) -> tuple[BoundJoin, ...]:
        out: list[BoundJoin] = []
        for ordinal, j in enumerate(node.args.get("joins") or []):
            if not isinstance(j, exp.Join):
                continue
            if _supports_projection_stars(self._query_contract) and (
                j.args.get("using") or str(j.args.get("method") or "").upper() == "NATURAL"
            ):
                # BoundJoin 只有 ON 表达式；吞掉 USING/NATURAL 会让不同连接语义
                # 合并成同一身份，也会令 unqualified * 的列顺序/去重规则失真。
                raise _fail(
                    SQL_UNSUPPORTED_STATEMENT,
                    "暂不编码 JOIN USING/NATURAL",
                )
            target = j.this
            if isinstance(target, exp.Subquery):
                if not _supports_query_sources(self._query_contract):
                    raise _fail(
                        SQL_UNSUPPORTED_STATEMENT,
                        "JOIN derived source 超出当前 query contract",
                    )
                self._register_derived(
                    self._tables, target, scope_label=f"join/{ordinal}"
                )
            elif isinstance(target, exp.Table):
                if (
                    _supports_query_sources(self._query_contract)
                    and target.name.casefold() in self._cte_definitions
                ):
                    self._register_cte_source(
                        self._tables, target, scope_label=f"join/{ordinal}"
                    )
                else:
                    self._register_table(self._tables, target)
            else:
                raise _fail(SQL_UNSUPPORTED_STATEMENT, "JOIN 目标不是受支持的 table source")
            side = str(j.args.get("side") or "").upper()
            kind = str(j.args.get("kind") or "").upper()
            if "LEFT" in (side + kind):
                join_type = "LEFT"
            elif "CROSS" in (side + kind):
                if not _supports_query_extensions(self._query_contract):
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, "不支持 CROSS product")
                join_type = "CROSS"
            elif "RIGHT" in (side + kind) or "FULL" in (side + kind):
                raise _fail(SQL_UNSUPPORTED_STATEMENT, f"不支持的 JOIN 类型: {side} {kind}")
            else:
                join_type = "INNER"
            on = None
            on_node = j.args.get("on")
            if on_node is not None:
                if join_type == "CROSS":
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, "CROSS product 不接受 ON")
                on = self._expr(on_node)
            join_source: BoundTable | BoundDerivedTable
            if isinstance(target, exp.Subquery):
                alias = target.alias
                if alias:
                    scope_path = self._tables[alias.casefold()]
                else:
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, "JOIN derived source 必须有别名")
                join_source = self._derived_scopes[scope_path]
            elif target.name.casefold() in self._cte_definitions:
                alias = target.alias or target.name
                scope_path = self._tables[alias.casefold()]
                join_source = self._derived_scopes[scope_path]
            else:
                rid = self._resolve_relation_id(target.name)
                assert rid is not None
                join_source = BoundTable(relation_id=rid, alias=target.alias)
            out.append(
                BoundJoin(
                    table=join_source,
                    join_type=join_type,
                    on=on,
                )
            )
        return tuple(out)

    def _projections(self, node: exp.Select) -> tuple[BoundProjection, ...]:
        out: list[BoundProjection] = []
        for e in node.expressions:
            alias = e.alias if isinstance(e, exp.Alias) else None
            projection = e.this if isinstance(e, exp.Alias) else e
            is_projection_star = (
                isinstance(projection, exp.Star)
                or isinstance(projection, exp.Column)
                and isinstance(projection.this, exp.Star)
            )
            if is_projection_star and not _supports_projection_stars(
                self._query_contract
            ) and isinstance(projection, exp.Column) and projection.table:
                # 旧契约不认识 qualifier.*；不能把 SQL 语法误报成缺列，否则 pool
                # 可能按普通 schema 错误静默丢弃一条实际可执行候选。
                raise _fail(
                    SQL_UNSUPPORTED_STATEMENT,
                    "不支持限定 projection star",
                )
            if is_projection_star and _supports_projection_stars(
                self._query_contract
            ):
                if alias is not None:
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, "projection star 不接受输出 alias")
                out.extend(self._expand_projection_star(projection, node))
                continue
            if isinstance(e, exp.Alias):
                bound_expr = self._expr(e.this)
                if isinstance(bound_expr, BoundScalarSubquery) and not _supports_query_tree(
                    self._query_contract
                ):
                    bound_expr = BoundUnsupported(
                        CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED,
                        "scalar AVG successor 仅允许出现在谓词比较 operand",
                    )
                out.append(BoundProjection(expr=bound_expr, alias=alias))
            else:
                bound_expr = self._expr(e)
                if isinstance(bound_expr, BoundScalarSubquery) and not _supports_query_tree(
                    self._query_contract
                ):
                    bound_expr = BoundUnsupported(
                        CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED,
                        "scalar AVG successor 仅允许出现在谓词比较 operand",
                    )
                out.append(BoundProjection(expr=bound_expr, alias=None))
        return tuple(out)

    def _expand_projection_star(
        self, projection: exp.Expression, select: exp.Select
    ) -> tuple[BoundProjection, ...]:
        """以 frozen catalog/source order 展开 SELECT star 为显式列或 derived slots。

        别名只用于解析当前作用域，不进入投影身份；重复 alias 或 USING/NATURAL
        可能改变列来源/输出去重时 fail-closed，防止把不等价候选合并。
        """
        if isinstance(projection, exp.Star):
            star = projection
            qualifier = None
        else:
            assert isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)
            if projection.args.get("db") is not None or projection.args.get("catalog") is not None:
                raise _fail(SQL_UNSUPPORTED_STATEMENT, "跨 schema/database 的 projection star 超域")
            star = projection.this
            qualifier = projection.table or None
        if any(star.args.get(key) for key in ("except_", "replace", "rename")):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "带 EXCEPT/REPLACE/RENAME 的 star 超域")
        if qualifier is None:
            for join in select.args.get("joins") or []:
                if isinstance(join, exp.Join) and (
                    join.args.get("using")
                    or str(join.args.get("method") or "").upper() == "NATURAL"
                ):
                    raise _fail(
                        SQL_UNSUPPORTED_STATEMENT,
                        "unqualified star 遇到 USING/NATURAL 列合并规则",
                    )
            sources = [source_id for _, source_id in self._source_order]
            if not sources:
                raise _fail(SQL_UNSUPPORTED_STATEMENT, "无 FROM 来源时不能展开 projection star")
        else:
            sources = self._alias_sources.get(qualifier.casefold(), [])
            if not sources:
                raise _fail(
                    UNRESOLVED_RELATION,
                    f"projection star 来源不在 FROM 作用域: {qualifier!r}",
                )
            if len(sources) != 1:
                raise _fail(
                    AMBIGUOUS_COLUMN,
                    f"重复 alias 下的 projection star 歧义: {qualifier!r}",
                )

        expanded: list[BoundProjection] = []
        for source_id in sources:
            derived = self._derived_scopes.get(source_id)
            if derived is not None:
                expressions: tuple[BoundExpr, ...] = tuple(
                    BoundDerivedColumn(derived.scope_path, ordinal)
                    for ordinal in range(len(derived.output_names))
                )
            else:
                relation = self._catalog.relations.get(source_id)
                if relation is None:
                    raise _fail(UNRESOLVED_RELATION, f"projection star 来源未注册: {source_id!r}")
                expressions = tuple(
                    BoundColumn(source_id, column_id) for column_id in relation.columns
                )
            expanded.extend(BoundProjection(expr=expr, alias=None) for expr in expressions)
        return tuple(expanded)

    def _where(self, node: exp.Select) -> BoundExpr | None:
        where = node.args.get("where")
        if where is None:
            return None
        return self._expr(where.this if isinstance(where, exp.Where) else where)

    def _having(self, node: exp.Select) -> BoundExpr | None:
        """保留 HAVING 所属 SELECT block，避免与 WHERE 合并后丢失聚合阶段。"""
        having = node.args.get("having")
        if having is None:
            return None
        if not _supports_query_shapes(self._query_contract):
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "HAVING 超出当前 query contract")
        return self._expr(having.this if isinstance(having, exp.Having) else having)

    def _group_by(self, node: exp.Select) -> tuple[BoundExpr, ...]:
        group = node.args.get("group")
        if group is None:
            return ()
        exprs = group.args.get("expressions") or []
        return tuple(self._expr(e) for e in exprs)

    def _order_by(
        self,
        node: exp.Select,
        *,
        projections: tuple[BoundProjection, ...],
    ) -> tuple[BoundOrderItem, ...]:
        """绑定 ORDER BY 各项为 BoundOrderItem。

        sqlglot 在 postgres 方言下已规范化：
        - ``desc`` 为 None/False → ASC，True → DESC；
        - ``nulls_first`` 为 True → FIRST，False → LAST（含方言隐式默认）。
        本层只做封闭枚举映射，不重新猜测方言默认。
        """
        order = node.args.get("order")
        if order is None:
            return ()
        exprs = order.args.get("expressions") or []
        out: list[BoundOrderItem] = []
        for e in exprs:
            ordered = e.this if isinstance(e, exp.Ordered) else e
            alias_matches = (
                [
                    projection
                    for projection in projections
                    if projection.alias is not None
                    and projection.alias.lower() == ordered.name.lower()
                ]
                if isinstance(ordered, exp.Column) and not ordered.table
                else []
            )
            if len(alias_matches) > 1:
                raise _fail(
                    AMBIGUOUS_COLUMN,
                    f"ORDER BY 投影别名不唯一: {ordered.name!r}",
                )
            # ORDER BY 输出别名按其投影表达式绑定，使身份不依赖展示名。
            inner = alias_matches[0].expr if alias_matches else self._expr(ordered)
            desc = bool(e.args.get("desc")) if isinstance(e, exp.Ordered) else False
            nulls_first = (
                bool(e.args.get("nulls_first")) if isinstance(e, exp.Ordered) else False
            )
            out.append(
                BoundOrderItem(
                    expr=inner,
                    direction="DESC" if desc else "ASC",
                    null_order="FIRST" if nulls_first else "LAST",
                )
            )
        return tuple(out)

    def _row_limit(self, node: exp.Select) -> tuple[int | None, int | None, bool]:
        """非负整数 LIMIT/OFFSET 与 FETCH 只保存可无损复现的值。

        任意表达式、参数、负值或 dialect-specific limit modifier 都不能猜测成行数；
        丢失它们会改变 scalar cardinality 和 derived-table 的结果语义。FETCH 只接受
        ROWS FIRST/NEXT，并明确保存 WITH TIES。
        """
        with_ties = False

        def count(value: exp.Expression | None, clause: str) -> int | None:
            if value is None:
                return None
            if isinstance(value, exp.Fetch):
                if self._query_contract != QUERY_CONTRACT_CADENCE:
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, "FETCH 不属于当前支持的 query contract")
                options = value.args.get("limit_options")
                if (
                    value.args.get("direction") not in {"FIRST", "NEXT"}
                    or not isinstance(options, exp.LimitOptions)
                    or options.args.get("rows") is not True
                    or options.args.get("percent") is True
                ):
                    raise _fail(SQL_UNSUPPORTED_STATEMENT, "FETCH 只支持 ROWS FIRST/NEXT")
                nonlocal with_ties
                with_ties = options.args.get("with_ties") is True
                value = value.args.get("count")
            if isinstance(value, (exp.Limit, exp.Offset)):
                value = value.args.get("expression") or value.this
            if (
                not isinstance(value, exp.Literal)
                or value.is_string
                or not str(value.this).isdigit()
            ):
                raise _fail(
                    SQL_UNSUPPORTED_STATEMENT,
                    f"{clause} 必须是非负整数 literal",
                )
            return int(str(value.this))

        limit = count(node.args.get("limit"), "LIMIT")
        offset = count(node.args.get("offset"), "OFFSET")
        if with_ties and node.args.get("order") is None:
            raise _fail(SQL_UNSUPPORTED_STATEMENT, "FETCH WITH TIES 需要 ORDER BY")
        return limit, offset, with_ties

    # -- 表达式绑定 -----------------------------------------------------
    def _membership(self, node: exp.In, *, negated: bool) -> BoundExpr:
        """独立绑定 membership 来源；局部 scope 不可借用外层列。

        当前范围接受单列单表 SELECT；带 JOIN 时要求恰有一条 INNER 等值 JOIN。
        内层过滤与 NULL 语义必须保留，不能将连接查询压成不完整集合来源。
        """
        operand = self._expr(node.this)
        source = node.args.get("query")
        if source is None:
            return BoundIn(operand, tuple(self._expr(v) for v in node.expressions), negated)
        reason = CADENCE_PREDICATE_SOURCE_UNBOUND
        if node.expressions or any(
            v for k, v in node.args.items() if k not in {"this", "query", "expressions"}
        ):
            return BoundUnsupported(reason, "membership 来源混用或修饰超域")
        if isinstance(source, exp.Subquery):
            if any(v for k, v in source.args.items() if k != "this"):
                return BoundUnsupported(reason, "子查询包装超域")
            source = source.this
        if _supports_query_tree(self._query_contract):
            self._require_query_tree_depth()
            if not isinstance(source, (exp.Select, exp.SetOperation)):
                return BoundUnsupported(reason, "membership 来源不是 SELECT/query tree")
            # 每个 local query 独立嵌入所属 expression；不使用遍历计数作身份，
            # 否则 AND/OR canonical 排序后，交换等价条件会改变子查询 scope path。
            scope_path = f"{self._scope_path}/local"
            local = _Binder(
                self._catalog,
                query_contract=self._query_contract,
                dialect=self._dialect,
                scope_path=scope_path,
                scope="LOCAL",
                query_depth=self._query_depth + 1,
                cte_definitions=self._cte_definitions,
            )
            try:
                query = local.bind(source)
            except ParseBindError as exc:
                if exc.reason_code == CADENCE_PREDICATE_SUBQUERY_DEPTH_UNSUPPORTED:
                    raise
                if self._query_references_outer(source, local):
                    raise _fail(
                        CADENCE_PREDICATE_CORRELATED_SUBQUERY,
                        "membership query 引用外层列",
                    ) from exc
                return BoundUnsupported(reason, f"membership query 未闭合: {exc.reason_code}")
            if _query_projection_count(query) != 1:
                return BoundUnsupported(reason, "membership query 必须产生一列")
            return BoundIn(operand, (), negated, BoundSubquery(query))
        if not isinstance(source, exp.Select):
            return BoundUnsupported(reason, "来源不是单 SELECT")
        # 在扫描列之前拒绝嵌套 scope：find_all 会进入内层 SELECT，若直接使用
        # 局部/外层 scope 解析其中的列，会伪造关联引用或丢失真实来源 lineage。
        if any(
            isinstance(child, (exp.Select, exp.Subquery, exp.SetOperation))
            for child in source.walk()
            if child is not source
        ):
            return BoundUnsupported(reason, "子查询嵌套查询超域")
        from_node = source.args.get("from_")
        if not isinstance(from_node, exp.From):
            return BoundUnsupported(reason, "子查询必须直接引用普通表")
        table = from_node.this
        if not isinstance(table, exp.Table):
            return BoundUnsupported(reason, "子查询必须直接引用普通表")
        if any(v for k, v in table.args.items() if k not in {"this", "alias"}):
            return BoundUnsupported(reason, "子查询表限定/修饰超域")
        if not isinstance(table.this, exp.Identifier):
            return BoundUnsupported(reason, "子查询来源不是普通表")
        if table.args.get("alias") is not None and table.args["alias"].args.get("columns"):
            return BoundUnsupported(reason, "列别名列表超域")
        join_nodes = source.args.get("joins") or []
        if join_nodes and (
            not _supports_query_shapes(self._query_contract)
            or len(join_nodes) != 1
        ):
            return BoundUnsupported(
                reason, "membership JOIN 仅允许单条 INNER equijoin"
            )
        local = _Binder(
            self._catalog,
            query_contract=self._query_contract,
            dialect=self._dialect,
            query_depth=self._query_depth + 1,
            cte_definitions=self._cte_definitions,
        )
        try:
            local._register_table(local._tables, table)
        except ParseBindError:
            return BoundUnsupported(reason, "子查询表未唯一绑定")
        if any(v for k, v in from_node.args.items() if k != "this"):
            return BoundUnsupported(reason, "membership FROM 修饰超域")
        for join in join_nodes:
            target = join.this if isinstance(join, exp.Join) else None
            if not isinstance(target, exp.Table) or any(
                v for k, v in target.args.items() if k not in {"this", "alias"}
            ):
                return BoundUnsupported(reason, "membership JOIN 目标必须是普通表")
            side = str(join.args.get("side") or "").upper()
            kind = str(join.args.get("kind") or "").upper()
            if side or kind not in {"", "INNER"}:
                return BoundUnsupported(reason, "membership JOIN 必须为 INNER")
            if any(
                v for k, v in join.args.items() if k not in {"this", "on", "side", "kind", "pivots"}
            ):
                return BoundUnsupported(reason, "membership JOIN 修饰超域")
        try:
            joins = local._joins(source) if join_nodes else ()
        except ParseBindError:
            return BoundUnsupported(reason, "membership JOIN 关系或条件未唯一绑定")
        # 先审查本层引用，包括即将拒绝的 WHERE。局部 alias 遮蔽外层同名
        # alias，即使局部缺少该列，也不能回退到外层同名 alias 来制造关联。
        for column in source.find_all(exp.Column):
            if any(v for k, v in column.args.items() if k not in {"this", "table"}):
                return BoundUnsupported(reason, "子查询列限定/修饰超域")
            try:
                local._column(column)
            except ParseBindError:
                if column.table and column.table.lower() in local._tables:
                    return BoundUnsupported(reason, "局部限定列未唯一绑定")
                try:
                    self._column(column)
                except ParseBindError:
                    return BoundUnsupported(reason, "子查询列未唯一绑定")
                return BoundUnsupported(CADENCE_PREDICATE_CORRELATED_SUBQUERY, "禁止 outer 引用")
        allowed_source_args = {"expressions", "from_", "where"}
        if join_nodes:
            allowed_source_args.add("joins")
        if (
            any(v for k, v in source.args.items() if k not in allowed_source_args)
            or len(source.expressions) != 1
        ):
            return BoundUnsupported(reason, "子查询含过滤/组合/排序等超域结构")
        projection = source.expressions[0]
        if isinstance(projection, exp.Alias):
            if any(v for k, v in projection.args.items() if k not in {"this", "alias"}):
                return BoundUnsupported(reason, "子查询投影别名修饰超域")
            projection = projection.this
        if not isinstance(projection, exp.Column) or projection.is_star:
            return BoundUnsupported(reason, "子查询投影必须为一个直接列")
        bound_projection = local._column(projection)
        if not isinstance(bound_projection, BoundColumn):
            return BoundUnsupported(reason, "membership 投影列未唯一绑定")
        bound_membership_column = bound_projection
        where = local._where(source)
        if join_nodes and (
            len(joins) != 1
            or joins[0].join_type != "INNER"
            or not isinstance(joins[0].on, BoundBinary)
            or joins[0].on.op != "eq"
            or not isinstance(joins[0].on.left, BoundColumn)
            or not isinstance(joins[0].on.right, BoundColumn)
            or joins[0].on.left.relation_id == joins[0].on.right.relation_id
        ):
            return BoundUnsupported(reason, "membership JOIN 必须是两关系之间的一条等值 INNER JOIN")
        if not join_nodes and where is not None and not (
            isinstance(where, BoundIsNull)
            and where.negated
            and isinstance(where.operand, BoundColumn)
            and where.operand == bound_membership_column
        ):
            return BoundUnsupported(
                CADENCE_PREDICATE_SUBQUERY_FILTER_UNSUPPORTED,
                "仅允许投影列 IS NOT NULL 过滤",
            )
        rid = local._tables[table.alias.lower() if table.alias else table.name.lower()]
        if bound_membership_column.relation_id != rid:
            return BoundUnsupported(reason, "joined membership 投影必须来自 FROM 主关系")
        query = BoundQuery(
            projections=(BoundProjection(bound_membership_column, None),),
            from_table=BoundTable(rid, None),
            joins=joins, where=where, group_by=(), order_by=(), ctes=(), distinct=False,
        )
        return BoundIn(operand, (), negated, BoundSubquery(query))

    def _scalar_avg_subquery(self, node: exp.Subquery) -> BoundExpr:
        """绑定支持的单表标量聚合子查询。"""
        if _supports_query_tree(self._query_contract):
            self._require_query_tree_depth()
            if any(value for key, value in node.args.items() if key != "this"):
                return BoundUnsupported(
                    CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED,
                    "scalar query 包装修饰超域",
                )
            source = node.this
            if not isinstance(source, (exp.Select, exp.SetOperation)):
                return BoundUnsupported(
                    CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED,
                    "scalar query 不是 SELECT/query tree",
                )
            scope_path = f"{self._scope_path}/local"
            local = _Binder(
                self._catalog,
                query_contract=self._query_contract,
                dialect=self._dialect,
                scope_path=scope_path,
                scope="LOCAL",
                query_depth=self._query_depth + 1,
                cte_definitions=self._cte_definitions,
                outer_scopes=(
                    (self._outer_scope_snapshot(), *self._outer_scopes)
                    if _supports_correlated_scalar_queries(self._query_contract)
                    else ()
                ),
            )
            try:
                query = local.bind(source)
            except ParseBindError as exc:
                if exc.reason_code == CADENCE_PREDICATE_SUBQUERY_DEPTH_UNSUPPORTED:
                    raise
                if self._query_references_outer(source, local):
                    return BoundUnsupported(
                        CADENCE_PREDICATE_SCALAR_SUBQUERY_CORRELATED,
                        "scalar query 引用外层列",
                    )
                return BoundUnsupported(
                    CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED,
                    f"scalar query 未闭合: {exc.reason_code}",
                )
            if _query_projection_count(query) != 1:
                return BoundUnsupported(
                    CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED,
                    "scalar query 必须产生一列",
                )
            return BoundScalarSubquery(query)
        if not _supports_query_shapes(self._query_contract):
            return BoundUnsupported(
                CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED,
                "当前查询范围不支持 scalar aggregate query",
            )
        if any(value for key, value in node.args.items() if key != "this"):
            return BoundUnsupported(
                CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED, "scalar query 包装修饰超域"
            )
        source = node.this
        if not isinstance(source, exp.Select) or any(
            isinstance(child, (exp.Select, exp.Subquery, exp.SetOperation))
            for child in source.walk()
            if child is not source
        ):
            return BoundUnsupported(
                CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED, "scalar query 必须是单层 SELECT"
            )
        if any(
            value for key, value in source.args.items() if key not in {"expressions", "from_"}
        ):
            return BoundUnsupported(
                CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED,
                "scalar aggregate 禁止 JOIN/WHERE/GROUP/ORDER/DISTINCT 等修饰",
            )
        from_node = source.args.get("from_")
        if not isinstance(from_node, exp.From):
            return BoundUnsupported(
                CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED,
                "scalar aggregate 必须直接引用普通表",
            )
        table = from_node.this
        if (
            not isinstance(table, exp.Table)
            or not isinstance(table.this, exp.Identifier)
            or any(value for key, value in table.args.items() if key not in {"this", "alias"})
            or (table.args.get("alias") is not None and table.args["alias"].args.get("columns"))
            or any(value for key, value in from_node.args.items() if key != "this")
            or len(source.expressions) != 1
        ):
            return BoundUnsupported(
                CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED,
                "scalar aggregate 必须是单表、单投影 SELECT",
            )
        projection = source.expressions[0]
        is_avg_column = isinstance(projection, exp.Avg) and isinstance(
            projection.this, exp.Column
        )
        is_count_star = isinstance(projection, exp.Count) and isinstance(
            projection.this, exp.Star
        )
        if not is_avg_column and not is_count_star:
            return BoundUnsupported(
                CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED,
                "scalar query 仅允许 AVG(直接列) 或 COUNT(*)",
            )
        local = _Binder(
            self._catalog,
            query_contract=self._query_contract,
            dialect=self._dialect,
            query_depth=self._query_depth + 1,
            cte_definitions=self._cte_definitions,
        )
        try:
            local._register_table(local._tables, table)
        except ParseBindError:
            return BoundUnsupported(
                CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED,
                "scalar aggregate 来源未唯一绑定",
            )
        for column in source.find_all(exp.Column):
            try:
                local._column(column)
            except ParseBindError:
                try:
                    self._column(column)
                except ParseBindError:
                    return BoundUnsupported(
                        CADENCE_PREDICATE_SCALAR_SUBQUERY_UNSUPPORTED,
                        "scalar aggregate 列未唯一绑定",
                    )
                return BoundUnsupported(
                    CADENCE_PREDICATE_SCALAR_SUBQUERY_CORRELATED,
                    "scalar aggregate 禁止引用外层查询列",
                )
        return BoundScalarSubquery(local._bind_select(source))

    def _not_exists(self, node: exp.Exists) -> BoundExpr:
        """绑定最小 ``NOT EXISTS`` correlation，不改写成其它 negative form。

        只接受 ``SELECT 1 FROM <single table> WHERE inner_col = outer_col``。
        该收窄足以覆盖 poker_player 的候选，同时拒绝额外过滤、OR、嵌套和
        多表形状；拒绝时保留稳定 reason code，避免把未知存在性当空集合。
        """
        reason = CADENCE_PREDICATE_NOT_EXISTS_SHAPE_UNSUPPORTED
        source = node.this
        if isinstance(source, exp.Subquery):
            if any(v for k, v in source.args.items() if k != "this"):
                return BoundUnsupported(reason, "EXISTS 包装含额外修饰")
            source = source.this
        if not isinstance(source, exp.Select):
            return BoundUnsupported(reason, "EXISTS 来源不是单 SELECT")
        if any(
            isinstance(child, (exp.Select, exp.Subquery, exp.SetOperation))
            for child in source.walk()
            if child is not source
        ):
            return BoundUnsupported(reason, "EXISTS 含嵌套查询")
        if any(v for k, v in source.args.items() if k not in {"expressions", "from_", "where"}):
            return BoundUnsupported(reason, "EXISTS 含过滤以外的查询修饰")
        from_node = source.args.get("from_")
        table = from_node.this if isinstance(from_node, exp.From) else None
        if not isinstance(table, exp.Table):
            return BoundUnsupported(reason, "EXISTS 必须直接引用单表")
        if any(v for k, v in table.args.items() if k not in {"this", "alias"}):
            return BoundUnsupported(reason, "EXISTS 来源表含额外修饰")
        if len(source.expressions) != 1:
            return BoundUnsupported(reason, "EXISTS 投影必须只有一项")
        projection = source.expressions[0]
        if (
            not isinstance(projection, exp.Literal)
            or projection.is_string
            or projection.this != "1"
        ):
            return BoundUnsupported(reason, "EXISTS 仅允许 SELECT 1")

        local = _Binder(
            self._catalog,
            query_contract=self._query_contract,
            dialect=self._dialect,
        )
        try:
            local._register_table(local._tables, table)
        except ParseBindError:
            return BoundUnsupported(reason, "EXISTS 来源表未唯一绑定")
        where_node = source.args.get("where")
        where_expr = where_node.this if isinstance(where_node, exp.Where) else where_node
        if not isinstance(where_expr, exp.EQ):
            return BoundUnsupported(reason, "EXISTS 必须是单一等值 correlation")
        left, right = where_expr.left, where_expr.right
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            return BoundUnsupported(reason, "EXISTS correlation 两侧必须是列")

        def local_column(column: exp.Column) -> BoundColumn | None:
            try:
                bound = local._column(column)
            except ParseBindError:
                return None
            return bound if isinstance(bound, BoundColumn) else None

        def outer_column(column: exp.Column) -> BoundColumn | None:
            try:
                bound = self._column(column)
            except ParseBindError:
                return None
            return bound if isinstance(bound, BoundColumn) else None

        left_inner, right_inner = local_column(left), local_column(right)
        left_outer, right_outer = outer_column(left), outer_column(right)
        if left_inner is not None and right_outer is not None and right_inner is None:
            inner_column, bound_outer_column = left_inner, right_outer
        elif right_inner is not None and left_outer is not None and left_inner is None:
            inner_column, bound_outer_column = right_inner, left_outer
        else:
            return BoundUnsupported(
                CADENCE_PREDICATE_NOT_EXISTS_CORRELATION_UNBOUND,
                "correlation 必须恰有一侧为局部列、一侧为外层列",
            )
        rid = next(iter(local._tables.values()))
        return BoundNotExistsCorrelated(
            BoundTable(rid, None), inner_column, bound_outer_column
        )

    def _expr(self, node: exp.Expression) -> BoundExpr:
        if isinstance(node, exp.Paren):
            # Paren 是 SQL 语法分组，不是独立语义节点；递归绑定其内部 AST，保留 AND/OR 树的优先级。
            # 若把包装节点当成 unsupported，括号条件会丢失可投影的谓词结构。
            return self._expr(node.this)
        if isinstance(node, exp.Column):
            return self._column(node)
        if isinstance(node, exp.Subquery):
            return self._scalar_avg_subquery(node)
        if isinstance(node, exp.Literal):
            # 字面量分类：字符串走 STRING；数值按是否含小数点区分
            # INTEGER / DECIMAL。sqlglot 把数值存为原始文本字符串，保留以保
            # canonical 确定性（不在本层做尾随零规范化——那是 typed producer 职责）。
            if node.is_string:
                return BoundLiteral(typed_tag="STRING", canonical=node.this)
            text = node.this
            if "." in text:
                return BoundLiteral(typed_tag="DECIMAL", canonical=text)
            return BoundLiteral(typed_tag="INTEGER", canonical=text)
        if isinstance(node, exp.Null):
            return BoundLiteral(typed_tag="NULL", canonical=None)
        if isinstance(node, exp.Boolean):
            return BoundLiteral(typed_tag="BOOLEAN", canonical="true" if node.this else "false")
        if isinstance(node, exp.Window):
            return self._window(node)
        if isinstance(node, exp.Cast):
            return self._cast_to_literal(node)
        if isinstance(node, exp.DataType):
            # CAST 目标类型等类型名：保真为 STRING 字面量，避免误判 unsupported。
            return BoundLiteral(typed_tag="STRING", canonical=str(node.this))
        if isinstance(node, exp.Star):
            return BoundStar()
        if isinstance(node, exp.Anonymous):
            args = tuple(self._expr(a) for a in node.expressions)
            return BoundFunc(name=str(node.this).lower(), args=args)
        if isinstance(node, exp.Exists):
            return BoundUnsupported(
                CADENCE_PREDICATE_NOT_EXISTS_SHAPE_UNSUPPORTED, "仅允许 NOT EXISTS"
            )
        if isinstance(node, exp.GroupConcat):
            if (
                not _supports_group_concat(self._query_contract)
                or self._dialect.casefold() != "sqlite"
            ):
                return BoundUnsupported(
                    "EXPR_GROUP_CONCAT_UNSUPPORTED",
                    "SQLite GROUP_CONCAT typed mapping 不受支持",
                )
            mapped = map_sqlite_group_concat_ast(node)
            if mapped is None:
                return BoundUnsupported(
                    "EXPR_GROUP_CONCAT_SHAPE_UNSUPPORTED",
                    "SQLGlot GroupConcat 节点未返回 mapping",
                )
            if not mapped.supported:
                return BoundUnsupported(
                    mapped.reason_code or "EXPR_GROUP_CONCAT_SHAPE_UNSUPPORTED",
                    mapped.detail,
                )
            return BoundFunc(
                name="group_concat",
                args=tuple(self._expr(argument) for argument in mapped.arguments),
                distinct=mapped.distinct,
            )
        if isinstance(node, exp.Func):
            distinct = isinstance(node.args.get("this"), exp.Distinct)
            return BoundFunc(
                name=_func_name(node),
                args=self._call_args(node),
                distinct=distinct,
            )
        if isinstance(node, exp.Between):
            return BoundBetween(
                operand=self._expr(node.this),
                lower=self._expr(node.args["low"]),
                upper=self._expr(node.args["high"]),
                negated=False,
            )
        if isinstance(node, exp.In):
            return self._membership(node, negated=False)
        if isinstance(node, exp.Is):
            if isinstance(node.expression, exp.Null):
                return BoundIsNull(operand=self._expr(node.this), negated=False)
            return BoundUnsupported("IS_OTHER", "IS 非 NULL 判定超出首版域")
        if isinstance(node, exp.Not):
            return self._not(node)
        if isinstance(node, exp.Neg):
            return BoundUnary(op="negate", operand=self._expr(node.this))
        if isinstance(node, exp.DPipe):
            if (
                not _supports_expression_extensions(self._query_contract)
                or node.args.get("safe") is not True
            ):
                return BoundUnsupported(
                    "EXPR_CONCAT_UNSUPPORTED",
                    "仅接受 sqlglot 标记为 safe 的标准 SQL || 连接",
                )
            return BoundBinary(
                op="concat", left=self._expr(node.left), right=self._expr(node.right)
            )
        if isinstance(node, exp.Binary):
            return BoundBinary(
                op=_binary_op(node), left=self._expr(node.left), right=self._expr(node.right)
            )
        return BoundUnsupported(
            f"EXPR_{type(node).__name__.upper()}", f"表达式未建模: {type(node).__name__}"
        )

    def _cast_to_literal(self, node: exp.Cast) -> BoundExpr:
        """CAST 规范化：把字面量 cast 收窄为 typed literal；其余 fail-closed。

        处理边界：
        - 本层只做"类型标签规范化"（Cast 节点 -> BoundLiteral 的 typed_tag），
          不校验日期/时间字符串是否合法——非法值（如 CAST('abc' AS DATE)）
          须送达 DuckDB 执行层由其裁决为 EXECUTION_ERROR，而非在绑定层
          短路为 UNSUPPORTED_AST。
        - DATE / TIMESTAMP：内层为 STRING 字面量即收窄为 DATE/TIMESTAMP typed
          literal，canonical 保留 sqlglot 原始文本（不自行格式化日期）。
        - DECIMAL / DOUBLE / FLOAT：内层数值字面量 -> DECIMAL；内层为
          NaN/Infinity 文本 -> BoundUnsupported（非有限数值不产生 canonical 字节）。
        - 内层是 Column（如 CAST(amount AS VARCHAR)）：当前绑定层不支持。
        - TIMESTAMPTZ / VARCHAR / TEXT / INTERVAL 等目标类型：当前绑定层不支持。
        """
        to_type = node.to
        target = to_type.this if isinstance(to_type, exp.DataType) else None
        inner = node.this

        if (
            target is exp.DataType.Type.INT
            and _supports_query_extensions(self._query_contract)
        ):
            return BoundCast(target_type="INT", operand=self._expr(inner))

        if (
            target in {
                exp.DataType.Type.DECIMAL,
                exp.DataType.Type.DOUBLE,
                exp.DataType.Type.FLOAT,
            }
            and _supports_expression_extensions(self._query_contract)
        ):
            # REAL 在 SQLite parser 中归一为 FLOAT；保留 parser 按目标 dialect
            # 解析出的类型标签，并把 CAST 节点纳入 identity，避免只保留内层列。
            return BoundCast(target_type=target.name, operand=self._expr(inner))

        if target is exp.DataType.Type.DATE:
            if isinstance(inner, exp.Literal) and inner.is_string:
                return BoundLiteral(typed_tag="DATE", canonical=inner.this)
            return BoundUnsupported("CAST_DATE_NON_LITERAL", "DATE cast 内层非字符串字面量")

        if target is exp.DataType.Type.TIMESTAMP:
            if isinstance(inner, exp.Literal) and inner.is_string:
                return BoundLiteral(typed_tag="TIMESTAMP", canonical=inner.this)
            return BoundUnsupported(
                "CAST_TIMESTAMP_NON_LITERAL", "TIMESTAMP cast 内层非字符串字面量"
            )

        if target in (
            exp.DataType.Type.DECIMAL,
            exp.DataType.Type.DOUBLE,
            exp.DataType.Type.FLOAT,
        ):
            # 数值 cast：内层数值字面量 -> DECIMAL；NaN/Infinity 文本一律拒绝。
            if isinstance(inner, exp.Literal) and not inner.is_string:
                return BoundLiteral(typed_tag="DECIMAL", canonical=inner.this)
            if (
                isinstance(inner, exp.Literal)
                and inner.is_string
                and inner.this in _NON_FINITE_NUMERIC_TEXTS
            ):
                return BoundUnsupported(
                    "NON_FINITE_NUMERIC", f"非有限数值字面量: {inner.this!r}"
                )
            return BoundUnsupported(
                "CAST_NUMERIC_INVALID", "数值 cast 内层非数值字面量或非有限值"
            )

        # TIMESTAMPTZ / VARCHAR / TEXT / INTERVAL 等当前不支持的目标类型。
        return BoundUnsupported(
            f"CAST_{target.name if target else 'UNKNOWN'}",
            f"CAST 目标类型不受支持: {target}",
        )

    def _window(self, node: exp.Window) -> BoundExpr:
        """按当前查询范围绑定有限窗口函数，并保留可无损表达的 ROWS/RANGE frame。"""
        if not _supports_query_extensions(self._query_contract):
            return BoundUnsupported("EXPR_WINDOW", "不支持该窗口表达式")
        frame_node = node.args.get("spec")
        allowed_args = {
            "this", "partition_by", "order", "spec", "alias", "over", "first",
        }
        if (
            any(key not in allowed_args and value is not None for key, value in node.args.items())
            or (
                frame_node is not None
                and self._query_contract != QUERY_CONTRACT_CADENCE
            )
            or node.args.get("alias") is not None
            or node.args.get("first") is not None
            or node.args.get("over") != "OVER"
        ):
            return BoundUnsupported("EXPR_WINDOW", "窗口 frame 或修饰不受支持")
        function = self._expr(node.this)
        aggregate_window = (
            isinstance(function, BoundFunc)
            and function.name in {"count", "sum", "avg", "min", "max"}
            and not function.distinct
            and len(function.args) == 1
            and not (
                isinstance(function.args[0], BoundStar) and function.name != "count"
            )
        )
        ranking_window = (
            isinstance(function, BoundFunc)
            and _supports_ranking_windows(self._query_contract)
            and function.name in {"rank", "dense_rank", "row_number"}
            and not function.distinct
            and function.args == ()
        )
        if not aggregate_window and not ranking_window:
            return BoundUnsupported("EXPR_WINDOW", "仅接受已声明的聚合窗口函数")
        if not isinstance(function, BoundFunc):
            return BoundUnsupported("EXPR_WINDOW", "窗口 function 未绑定为 registered function")
        partitions = tuple(self._expr(value) for value in (node.args.get("partition_by") or []))
        order_node = node.args.get("order")
        order_items: list[BoundOrderItem] = []
        if order_node is not None:
            if not isinstance(order_node, exp.Order) or any(
                key != "expressions" and value is not None
                for key, value in order_node.args.items()
            ):
                return BoundUnsupported("EXPR_WINDOW", "窗口 ORDER BY 修饰不受支持")
            for item in order_node.args.get("expressions") or []:
                if isinstance(item, exp.Ordered) and any(
                    key not in {"this", "desc", "nulls_first"} and value is not None
                    for key, value in item.args.items()
                ):
                    return BoundUnsupported("EXPR_WINDOW", "窗口排序修饰不受支持")
                ordered = item.this if isinstance(item, exp.Ordered) else item
                order_items.append(
                    BoundOrderItem(
                        expr=self._expr(ordered),
                        direction=(
                            "DESC"
                            if isinstance(item, exp.Ordered)
                            and item.args.get("desc")
                            else "ASC"
                        ),
                        null_order=(
                            "FIRST"
                            if isinstance(item, exp.Ordered)
                            and bool(item.args.get("nulls_first"))
                            else "LAST"
                        ),
                    )
                )
        frame = self._window_frame(frame_node) if frame_node is not None else None
        if isinstance(frame, BoundUnsupported):
            return frame
        if frame_node is not None and frame is None:
            return BoundUnsupported("EXPR_WINDOW", "窗口 frame 缺失或结构无效")
        return BoundWindow(function, partitions, tuple(order_items), frame)

    def _window_frame(self, node: exp.Expression) -> BoundWindowFrame | BoundUnsupported:
        """将可无损表达的窗口 frame 转为 typed IR。"""

        if not isinstance(node, exp.WindowSpec):
            return BoundUnsupported("EXPR_WINDOW", "窗口 frame 不是 WindowSpec")
        args = node.args
        unit = args.get("kind")
        if unit not in {"ROWS", "RANGE"}:
            return BoundUnsupported("EXPR_WINDOW", "window frame 仅支持 ROWS/RANGE")

        def make_bound(
            value: object,
            side: object,
            *,
            default_current_row: bool,
        ) -> BoundWindowFrameBound | BoundUnsupported:
            if value is None and default_current_row:
                return BoundWindowFrameBound("CURRENT_ROW", None, None)
            if value == "UNBOUNDED" and side in {"PRECEDING", "FOLLOWING"}:
                normalized_side = cast(Literal["PRECEDING", "FOLLOWING"], side)
                return BoundWindowFrameBound("UNBOUNDED", normalized_side, None)
            if value == "CURRENT ROW" and side is None:
                return BoundWindowFrameBound("CURRENT_ROW", None, None)
            if isinstance(value, exp.Expression) and side in {"PRECEDING", "FOLLOWING"}:
                offset = self._expr(value)
                if isinstance(offset, BoundLiteral):
                    normalized_side = cast(Literal["PRECEDING", "FOLLOWING"], side)
                    return BoundWindowFrameBound("OFFSET", normalized_side, offset)
            return BoundUnsupported("EXPR_WINDOW", "窗口 frame 边界无法无损表示")

        start = make_bound(args.get("start"), args.get("start_side"), default_current_row=False)
        end = make_bound(args.get("end"), args.get("end_side"), default_current_row=True)
        if isinstance(start, BoundUnsupported):
            return start
        if isinstance(end, BoundUnsupported):
            return end

        exclude = args.get("exclude")
        exclusion: Literal["NO_OTHERS", "CURRENT_ROW", "GROUP", "TIES"] = "NO_OTHERS"
        if exclude is not None:
            if not isinstance(exclude, exp.Var):
                return BoundUnsupported("EXPR_WINDOW", "窗口 frame exclusion 未注册")
            normalized_exclusion = str(exclude.this).upper().replace(" ", "_")
            if normalized_exclusion not in {"NO_OTHERS", "CURRENT_ROW", "GROUP", "TIES"}:
                return BoundUnsupported("EXPR_WINDOW", "窗口 frame exclusion 未注册")
            exclusion = cast(
                Literal["NO_OTHERS", "CURRENT_ROW", "GROUP", "TIES"],
                normalized_exclusion,
            )
        return BoundWindowFrame(unit, start, end, exclusion)


    def _not(self, node: exp.Not) -> BoundExpr:
        """NOT 解包：NOT BETWEEN / NOT IN / NOT IS NULL → 对应取反节点。"""
        inner = node.this
        if isinstance(inner, exp.Between):
            return BoundBetween(
                operand=self._expr(inner.this),
                lower=self._expr(inner.args["low"]),
                upper=self._expr(inner.args["high"]),
                negated=True,
            )
        if isinstance(inner, exp.In):
            return self._membership(inner, negated=True)
        if isinstance(inner, exp.Exists):
            return self._not_exists(inner)
        if isinstance(inner, exp.Is) and isinstance(inner.expression, exp.Null):
            return BoundIsNull(operand=self._expr(inner.this), negated=True)
        return BoundUnary(op="not", operand=self._expr(inner))

    def _call_args(self, node: exp.Func) -> tuple[BoundExpr, ...]:
        """按 sqlglot args 字典插入序收集函数参数（确定性）。

        跳过 None/bool 标志（如 count 的 big_int）；``Distinct`` 包装解包为
        内层表达式（distinct 标记由调用方单独读取）；命名标量参数（如
        time_to_str 的 format 字符串）转为 STRING 字面量，保证身份字节完整。
        """
        out: list[BoundExpr] = []
        cadence_timestamp_trunc = (
            self._query_contract == QUERY_CONTRACT_CADENCE
            and _func_name(node) == "timestamp_trunc"
        )
        for key, value in node.args.items():
            if value is None or isinstance(value, bool):
                continue
            if cadence_timestamp_trunc and key == "unit" and isinstance(value, exp.Var):
                # sqlglot 将 PostgreSQL DATE_TRUNC 的单位解析为 Var；仅在该函数的
                # unit 参数位置将其绑定为 STRING，避免方言语义变化。
                out.append(BoundLiteral("STRING", str(value.this).casefold()))
                continue
            if isinstance(value, exp.Distinct):
                inner = value.this
                if inner is not None:
                    out.append(self._expr(inner))
                else:
                    out.extend(self._expr(v) for v in value.expressions)
            elif isinstance(value, exp.Expression):
                out.append(self._expr(value))
            elif isinstance(value, (str, exp.Identifier)):
                out.append(BoundLiteral("STRING", str(value)))
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, exp.Expression):
                        out.append(self._expr(v))
                    elif isinstance(v, (str, exp.Identifier)):
                        out.append(BoundLiteral("STRING", str(v)))
            else:
                # 未知标量参数类型：以字符串保底投影，保持身份确定性与完整。
                out.append(BoundLiteral("STRING", str(value)))
        return tuple(out)

    def _column(self, node: exp.Column) -> BoundExpr:
        qualifier = node.table
        name = node.name
        if qualifier:
            # 限定列必须引用当前 FROM 作用域（别名 / 未别名表名 / CTE 名）；
            # 引用不在作用域的关系（含已起别名时用原表名）与 SQL 语义一致，fail-closed。
            key = qualifier.casefold()
            rid = self._tables.get(key)
            if rid is None:
                outer = self._qualified_outer_column(qualifier, name)
                if outer is not None:
                    return outer
                raise _fail(UNRESOLVED_RELATION, f"限定关系不在 FROM 作用域: {qualifier!r}")
            sources = self._alias_sources.get(key, [rid])
            if len(sources) > 1:
                qualified_matches = [
                    bound for source_id in sources
                    if (bound := self._column_from_source(source_id, name)) is not None
                ]
                if len(qualified_matches) > 1:
                    raise _fail(AMBIGUOUS_COLUMN, f"重复 alias 下的限定列歧义: {qualifier}.{name}")
                if not qualified_matches:
                    raise _fail(UNRESOLVED_COLUMN, f"重复 alias 下未解析列: {qualifier}.{name}")
                return qualified_matches[0]
            derived = self._derived_scopes.get(rid)
            if derived is not None:
                ordinals = _derived_output_ordinals(derived, name)
                if len(ordinals) > 1:
                    raise _fail(AMBIGUOUS_COLUMN, f"derived output 名称歧义: {name!r}")
                if not ordinals:
                    raise _fail(UNRESOLVED_COLUMN, f"未解析 derived output: {name!r}")
                return BoundDerivedColumn(derived.scope_path, ordinals[0])
            col = self._catalog.resolve_column(rid, name)
            if col is None:
                raise _fail(UNRESOLVED_COLUMN, f"未解析列: {rid}.{name!r}")
            return BoundColumn(relation_id=rid, column_id=col)
        # 未限定：必须在当前 FROM 作用域唯一。
        if _supports_sqlite_duplicate_aliases(
            self._query_contract, self._dialect
        ):
            alias_matches = [
                bound for sources in self._alias_sources.values()
                for source_id in sources
                if (bound := self._column_from_source(source_id, name)) is not None
            ]
            if len(alias_matches) > 1:
                raise _fail(AMBIGUOUS_COLUMN, f"歧义列: {name!r} 命中多个 FROM 来源")
            if alias_matches:
                return alias_matches[0]
            outer = self._unqualified_outer_column(name)
            if outer is not None:
                return outer
            if (
                self._dialect == "sqlite"
                and isinstance(node.this, exp.Identifier)
                and node.this.args.get("quoted") is True
            ):
                return BoundLiteral("STRING", name)
            raise _fail(UNRESOLVED_COLUMN, f"未解析列: {name!r}")
        column_matches: list[str] = []
        for rid in self._tables.values():
            derived = self._derived_scopes.get(rid)
            if derived is not None:
                ordinals = _derived_output_ordinals(derived, name)
                if len(ordinals) > 1:
                    raise _fail(AMBIGUOUS_COLUMN, f"derived output 名称歧义: {name!r}")
                found = bool(ordinals)
            else:
                found = self._catalog.resolve_column(rid, name) is not None
            if found:
                column_matches.append(rid)
        unique = sorted(set(column_matches))
        if not unique:
            outer = self._unqualified_outer_column(name)
            if outer is not None:
                return outer
            # SQLite 兼容历史 DQS：未解析的双引号 token 可作为字符串字面量。
            # 先查列名，才能保留 SQLite 对同名真实列的标识符优先规则。
            if (
                self._dialect == "sqlite"
                and isinstance(node.this, exp.Identifier)
                and node.this.args.get("quoted") is True
            ):
                return BoundLiteral("STRING", name)
            raise _fail(UNRESOLVED_COLUMN, f"未解析列: {name!r}")
        if len(unique) > 1:
            raise _fail(AMBIGUOUS_COLUMN, f"歧义列: {name!r} 命中 {unique}")
        derived = self._derived_scopes.get(unique[0])
        if derived is not None:
            ordinals = _derived_output_ordinals(derived, name)
            assert len(ordinals) == 1
            return BoundDerivedColumn(derived.scope_path, ordinals[0])
        col = self._catalog.resolve_column(unique[0], name)
        assert col is not None
        return BoundColumn(relation_id=unique[0], column_id=col)

    def _column_from_source(self, source_id: str, name: str) -> BoundExpr | None:
        """在一个物理/derived 来源中按 schema 唯一解析列名。"""
        derived = self._derived_scopes.get(source_id)
        if derived is not None:
            ordinals = _derived_output_ordinals(derived, name)
            if len(ordinals) > 1:
                raise _fail(AMBIGUOUS_COLUMN, f"derived output 名称歧义: {name!r}")
            return (
                BoundDerivedColumn(derived.scope_path, ordinals[0])
                if ordinals else None
            )
        column = self._catalog.resolve_column(source_id, name)
        return BoundColumn(source_id, column) if column is not None else None

    def _outer_scope_snapshot(
        self,
    ) -> dict[str, tuple[str | BoundDerivedTable, ...]]:
        """冻结当前 SELECT 可见的 alias 到物理 relation/derived source 映射。"""
        return {
            alias: tuple(
                self._derived_scopes.get(source_id, source_id)
                for source_id in self._alias_sources.get(alias, [source_id])
            )
            for alias, source_id in self._tables.items()
        }

    def _outer_column_from_source(
        self,
        source: str | BoundDerivedTable,
        name: str,
        depth: int,
    ) -> BoundOuterColumn | None:
        """将外层列名解析为 relation identity 或稳定 derived output slot。"""
        if isinstance(source, BoundDerivedTable):
            ordinals = _derived_output_ordinals(source, name)
            if len(ordinals) > 1:
                raise _fail(AMBIGUOUS_COLUMN, f"outer derived output 名称歧义: {name!r}")
            if not ordinals:
                return None
            return BoundOuterColumn(
                depth,
                scope_path=source.scope_path,
                projection_ordinal=ordinals[0],
            )
        column = self._catalog.resolve_column(source, name)
        if column is None:
            return None
        return BoundOuterColumn(depth, relation_id=source, column_id=column)

    def _qualified_outer_column(
        self, qualifier: str, name: str
    ) -> BoundOuterColumn | None:
        """在标量子查询中按 lexical alias 搜索外层 scope。"""
        if not _supports_correlated_scalar_queries(self._query_contract):
            return None
        key = qualifier.casefold()
        for depth, scope in enumerate(self._outer_scopes, start=1):
            sources = scope.get(key)
            if sources is not None:
                matches = [
                    result for source in sources
                    if (result := self._outer_column_from_source(source, name, depth)) is not None
                ]
                if len(matches) > 1:
                    raise _fail(
                        AMBIGUOUS_COLUMN,
                        f"重复 outer alias 下的限定列歧义: {qualifier}.{name}",
                    )
                if not matches:
                    raise _fail(UNRESOLVED_COLUMN, f"未解析 outer column: {qualifier}.{name}")
                return matches[0]
        return None

    def _unqualified_outer_column(self, name: str) -> BoundOuterColumn | None:
        """仅当当前查询没有该列时，按最近的外层作用域唯一解析未限定列。"""
        if not _supports_correlated_scalar_queries(self._query_contract):
            return None
        for depth, scope in enumerate(self._outer_scopes, start=1):
            matches = [
                ref for sources in scope.values() for source in sources
                if (ref := self._outer_column_from_source(source, name, depth)) is not None
            ]
            if len(matches) > 1:
                raise _fail(AMBIGUOUS_COLUMN, f"歧义 outer column: {name!r}")
            if matches:
                return matches[0]
        return None


def _func_name(node: exp.Func) -> str:
    """函数名归一化：sqlglot 具名函数类名转小写；Anonymous 用其名字。"""
    sql_name = node.sql_name()  # type: ignore[no-untyped-call]  # sqlglot 无 stub
    if sql_name:
        return str(sql_name).lower()
    return type(node).__name__.lower()


def _binary_op(node: exp.Binary) -> str:
    """二元运算名归一化（小写、稳定）。"""
    return type(node).__name__.lower()


# ---------------------------------------------------------------------------
# 唯一入口
# ---------------------------------------------------------------------------
def bind_sql(
    sql: str,
    catalog: SchemaCatalog,
    dialect: str = "postgres",
    expected_snapshot_hash: str | None = None,
    *,
    query_contract: str = QUERY_CONTRACT_CADENCE,
) -> BoundQuery:
    """解析 + 只读守卫 + 绑定；可选校验 catalog 快照哈希。

    参数：
        sql: 冻结候选 SQL 文本（不修改）。
        catalog: 数据库快照目录。
        dialect: sqlglot 读取方言（postgres/duckdb/sqlite 等）。
        expected_snapshot_hash: 声明快照哈希；提供时不一致即
            ``CATALOG_SNAPSHOT_HASH_MISMATCH``（B4 一致性前置校验）。
        query_contract: 当前 CADENCE SQL 查询范围。
    返回：
        确定性 BoundQuery（含已绑定列，供因子提取消费）。
    """
    if expected_snapshot_hash is not None:
        got = catalog.snapshot_hash()
        if got != expected_snapshot_hash:
            raise _fail(
                CATALOG_SNAPSHOT_HASH_MISMATCH,
                f"catalog 快照哈希 {got} != 声明 {expected_snapshot_hash}",
            )
    if query_contract != QUERY_CONTRACT_CADENCE:
        raise _fail(SQL_UNSUPPORTED_STATEMENT, f"未知 query contract: {query_contract}")
    select = ReadOnlyGuard(dialect=dialect).check(sql, query_contract=query_contract)
    return _Binder(
        catalog,
        query_contract=query_contract,
        dialect=dialect,
    ).bind(select)


def _query_projection_count(query: BoundQuery) -> int:
    """返回 SELECT/set tree 的左支输出列数；列数相容已由 binder 逐节点校验。"""
    if query.compound is not None:
        return _set_projection_count(query.compound)
    return len(query.projections)


def _set_projection_count(node: BoundSetOperation) -> int:
    branch = node.left
    return (
        _query_projection_count(branch)
        if isinstance(branch, BoundQuery)
        else _set_projection_count(branch)
    )


def _query_projection_names(query: BoundQuery) -> tuple[str | None, ...]:
    """按 SQL 规则用最左叶输出列名解析 derived table 的列引用。"""
    if query.compound is not None:
        left = query.compound.left
        return (
            _query_projection_names(left)
            if isinstance(left, BoundQuery)
            else _set_projection_names(left)
        )
    names: list[str | None] = []
    for projection in query.projections:
        if projection.alias is not None:
            names.append(projection.alias)
        elif isinstance(projection.expr, BoundColumn):
            names.append(projection.expr.column_id)
        elif isinstance(projection.expr, BoundDerivedColumn):
            source = _derived_source_by_scope(query, projection.expr.scope_path)
            ordinal = projection.expr.projection_ordinal
            names.append(
                source.output_names[ordinal]
                if source is not None and 0 <= ordinal < len(source.output_names)
                else None
            )
        else:
            names.append(None)
    return tuple(names)


def _derived_source_by_scope(
    query: BoundQuery, scope_path: str
) -> BoundDerivedTable | None:
    """通过绑定 scope 回查派生输出名，供 CTE 链和嵌套 derived 列名传递使用。

    派生列只有 slot ordinal；若这里丢失上游 output name，下一层 CTE 就无法
    解析外层引用，正确 SQL 会在结构身份生成前被排除。
    """
    if (
        isinstance(query.from_table, BoundDerivedTable)
        and query.from_table.scope_path == scope_path
    ):
        return query.from_table
    for join in query.joins:
        if (
            isinstance(join.table, BoundDerivedTable)
            and join.table.scope_path == scope_path
        ):
            return join.table
    return None


def _set_projection_names(node: BoundSetOperation) -> tuple[str | None, ...]:
    left = node.left
    return (
        _query_projection_names(left)
        if isinstance(left, BoundQuery)
        else _set_projection_names(left)
    )


def _query_contains_derived_table(query: BoundQuery) -> bool:
    """检测 derived subtree 中再嵌套 derived source，防止版本边界被递归扩张。"""
    if (
        isinstance(query.from_table, BoundDerivedTable)
        or any(isinstance(join.table, BoundDerivedTable) for join in query.joins)
    ):
        return True
    if query.compound is not None:
        return _set_contains_derived_table(query.compound)
    return any(_expr_contains_derived_query(expr) for expr in _query_expressions(query))


def _set_contains_derived_table(node: BoundSetOperation) -> bool:
    for branch in (node.left, node.right):
        if isinstance(branch, BoundQuery) and _query_contains_derived_table(branch):
            return True
        if isinstance(branch, BoundSetOperation) and _set_contains_derived_table(branch):
            return True
    return False


def _query_expressions(query: BoundQuery) -> Iterator[BoundExpr]:
    for projection in query.projections:
        yield projection.expr
    if query.where is not None:
        yield query.where
    if query.having is not None:
        yield query.having
    yield from query.group_by
    for order_item in query.order_by:
        yield order_item.expr
    for join in query.joins:
        if join.on is not None:
            yield join.on


def _expr_contains_derived_query(expr: BoundExpr) -> bool:
    for node in walk_bound_expr(expr):
        if isinstance(node, (BoundSubquery, BoundScalarSubquery)) and _query_contains_derived_table(
            node.query
        ):
            return True
    return False


def _derived_output_ordinals(table: BoundDerivedTable, name: str) -> list[int]:
    """按 casefold 输出名返回全部槽位，以便区分未解析和歧义。"""
    return [
        index
        for index, output_name in enumerate(table.output_names)
        if output_name is not None and output_name.casefold() == name.casefold()
    ]


def _derived_output_ordinal(table: BoundDerivedTable, name: str) -> int | None:
    matches = _derived_output_ordinals(table, name)
    return matches[0] if len(matches) == 1 else None


def _columns_at_current_query_level(node: exp.Expression) -> Iterator[exp.Column]:
    """遍历当前 SELECT/set level，跳过子查询，供 correlated scope 诊断使用。"""
    if isinstance(node, exp.Subquery):
        node = node.this

    def visit(current: exp.Expression, *, root: bool = False) -> Iterator[exp.Column]:
        if isinstance(current, exp.Subquery):
            return
        if isinstance(current, exp.Select) and not root:
            return
        if isinstance(current, exp.Column):
            yield current
        for value in current.args.values():
            if isinstance(value, exp.Expression):
                # Direct set branches share one query scope; nested scalar/IN subqueries do not.
                if isinstance(current, exp.SetOperation) and isinstance(value, exp.Select):
                    yield from visit(value, root=True)
                else:
                    yield from visit(value)
            elif isinstance(value, list):
                for child in value:
                    if isinstance(child, exp.Expression):
                        if isinstance(current, exp.SetOperation) and isinstance(child, exp.Select):
                            yield from visit(child, root=True)
                        else:
                            yield from visit(child)

    yield from visit(node, root=True)


def canonical_bound_query_tree(
    query: BoundQuery, *, allow_extended_expression_canonicalization: bool = False
) -> dict[str, Any]:
    """生成闭合 query-tree identity；扩展表达式能力只由对应 successor 显式开放。"""
    if query.compound is not None:
        return _canonical_set_query_tree(
            query.compound,
            scope=query.scope,
            allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
        )
    if query.ctes:
        raise ValueError("CADENCE_QUERY_TREE_CTE_UNSUPPORTED")
    from_source: dict[str, Any] | None
    if isinstance(query.from_table, BoundTable):
        from_source = {"kind": "RELATION", "relation_id": query.from_table.relation_id}
    elif isinstance(query.from_table, BoundDerivedTable):
        from_source = _canonical_derived_source(
            query.from_table,
            allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
        )
    else:
        from_source = None
    return {
        "kind": "SELECT",
        "scope": query.scope,
        "distinct": query.distinct,
        "projections": [
            {
                "ordinal": str(index),
                "expr": _query_expr_canonical(
                    item.expr,
                    allow_extended_expression_canonicalization=(
                        allow_extended_expression_canonicalization
                    ),
                ),
                "alias": item.alias,
            }
            for index, item in enumerate(query.projections)
        ],
        "from": from_source,
        "joins": [
            _canonical_join_source(
                join,
                allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
            )
            for join in query.joins
        ],
        "where": _query_expr_canonical(
            query.where,
            allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
        ) if query.where is not None else None,
        "group_by": [
            _query_expr_canonical(
                expr,
                allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
            )
            for expr in query.group_by
        ],
        "having": _query_expr_canonical(
            query.having,
            allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
        ) if query.having is not None else None,
        "order_by": [
            {
                "expr": _query_expr_canonical(
                    item.expr,
                    allow_extended_expression_canonicalization=(
                        allow_extended_expression_canonicalization
                    ),
                ),
                "direction": item.direction,
                "null_order": item.null_order,
            }
            for item in query.order_by
        ],
        "limit": str(query.limit) if query.limit is not None else None,
        "offset": str(query.offset) if query.offset is not None else None,
    }


def _canonical_derived_source(
    source: BoundDerivedTable,
    *,
    allow_extended_expression_canonicalization: bool,
) -> dict[str, Any]:
    """序列化 FROM/JOIN derived source，并保留输出 ordinal 与其 query 身份。"""
    return {
        "kind": "DERIVED",
        "scope_path": source.scope_path,
        "query": canonical_bound_query_tree(
            source.query,
            allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
        ),
        "output_names": list(source.output_names),
    }


def _canonical_join_source(
    join: BoundJoin,
    *,
    allow_extended_expression_canonicalization: bool,
) -> dict[str, Any]:
    """直接关系保持历史 canonical shape；derived target 增加显式 query source。"""
    target: dict[str, Any]
    if isinstance(join.table, BoundTable):
        target = {"relation_id": join.table.relation_id}
    else:
        target = {
            "source": _canonical_derived_source(
                join.table,
                allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
            )
        }
    target.update({
        "join_type": join.join_type,
        "on": _query_expr_canonical(
            join.on,
            allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
        ) if join.on is not None else None,
    })
    return target


def canonical_bound_expression(
    expr: BoundExpr, *, allow_extended_expression_canonicalization: bool = False
) -> dict[str, Any]:
    """返回闭合 typed expression canonical 值；扩展表达式规范化必须显式启用。"""
    return _query_expr_canonical(
        expr, allow_extended_expression_canonicalization=allow_extended_expression_canonicalization
    )


def _canonical_set_query_tree(
    node: BoundSetOperation,
    *,
    scope: Literal["GLOBAL", "LOCAL"],
    allow_extended_expression_canonicalization: bool = False,
) -> dict[str, Any]:
    return {
        "kind": "SET_OPERATION",
        "scope": scope,
        "operator": node.operator,
        "quantifier": node.quantifier,
        "left": _canonical_query_branch(
            node.left,
            allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
        ),
        "right": _canonical_query_branch(
            node.right,
            allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
        ),
    }


def _canonical_query_branch(
    branch: BoundQuery | BoundSetOperation,
    *,
    allow_extended_expression_canonicalization: bool = False,
) -> dict[str, Any]:
    return (
        canonical_bound_query_tree(
            branch,
            allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
        )
        if isinstance(branch, BoundQuery)
        else _canonical_set_query_tree(
            branch,
            scope="LOCAL",
            allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
        )
    )


def _query_expr_canonical(
    expr: BoundExpr, *, allow_extended_expression_canonicalization: bool = False
) -> dict[str, Any]:
    if isinstance(expr, BoundOuterColumn):
        if expr.scope_path is not None:
            return {
                "kind": "OUTER_DERIVED_OUTPUT_SLOT",
                "scope_depth": str(expr.scope_depth),
                "scope_path": expr.scope_path,
                "projection_ordinal": str(expr.projection_ordinal),
            }
        return {
            "kind": "OUTER_COLUMN",
            "scope_depth": str(expr.scope_depth),
            "relation_id": expr.relation_id,
            "column_id": expr.column_id,
        }
    if isinstance(expr, BoundDerivedColumn):
        return {
            "kind": "DERIVED_OUTPUT_SLOT",
            "scope_path": expr.scope_path,
            "projection_ordinal": str(expr.projection_ordinal),
        }
    if isinstance(expr, BoundIn):
        if expr.subquery is None and not expr.values:
            raise ValueError(CADENCE_PREDICATE_QUERY_EXPRESSION_UNSUPPORTED)
        result: dict[str, Any] = {
            "kind": "IN",
            "operand": _query_expr_canonical(
                expr.operand,
                allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
            ),
            "negated": expr.negated,
        }
        if expr.subquery is None:
            result["values"] = [
                _query_expr_canonical(
                    value,
                    allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
                )
                for value in expr.values
            ]
        else:
            if expr.values:
                raise ValueError(CADENCE_PREDICATE_SOURCE_UNBOUND)
            result["query"] = canonical_bound_query_tree(
                expr.subquery.query,
                allow_extended_expression_canonicalization=(
                    allow_extended_expression_canonicalization
                ),
            )
        return result
    if isinstance(expr, BoundScalarSubquery):
        return {
            "kind": "SCALAR_QUERY",
            "query": canonical_bound_query_tree(
                expr.query,
                allow_extended_expression_canonicalization=(
                    allow_extended_expression_canonicalization
                ),
            ),
            "cardinality_rule": "SQL_SCALAR_SUBQUERY",
        }
    if isinstance(expr, BoundCast):
        allowed_cast_types = {"INT"}
        if allow_extended_expression_canonicalization:
            allowed_cast_types.update({"DECIMAL", "DOUBLE", "FLOAT"})
        if expr.target_type not in allowed_cast_types:
            raise ValueError(CADENCE_PREDICATE_QUERY_EXPRESSION_UNSUPPORTED)
        return {
            "kind": "cast",
            "target_type": expr.target_type,
            "operand": _query_expr_canonical(
                expr.operand,
                allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
            ),
        }
    if isinstance(expr, BoundWindow):
        if expr.frame is not None:
            raise ValueError(CADENCE_PREDICATE_QUERY_EXPRESSION_UNSUPPORTED)
        if expr.function.name in {"rank", "dense_rank", "row_number"}:
            if expr.function.distinct or expr.function.args:
                raise ValueError(CADENCE_PREDICATE_QUERY_EXPRESSION_UNSUPPORTED)
            function_canonical = {
                "kind": "func",
                "name": expr.function.name,
                "args": [],
                "distinct": False,
            }
        else:
            function_canonical = _query_expr_canonical(
                expr.function,
                allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
            )
        return {
            "kind": "window",
            "function": function_canonical,
            "partition_by": [
                _query_expr_canonical(
                    item,
                    allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
                )
                for item in expr.partition_by
            ],
            "order_by": [
                {
                    "expr": _query_expr_canonical(
                        item.expr,
                        allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
                    ),
                    "direction": item.direction,
                    "null_order": item.null_order,
                }
                for item in expr.order_by
            ],
            "frame": None,
        }
    if isinstance(expr, BoundSubquery):
        return canonical_bound_query_tree(
            expr.query,
            allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
        )
    if isinstance(expr, BoundBinary):
        binary_operators = (
            _QUERY_TREE_EXTENDED_BINARY_OPERATORS
            if allow_extended_expression_canonicalization
            else _QUERY_TREE_BINARY_OPERATORS
        )
        if expr.op not in binary_operators:
            raise ValueError(CADENCE_PREDICATE_QUERY_EXPRESSION_UNSUPPORTED)
        return {
            "kind": "binary", "op": expr.op,
            "left": _query_expr_canonical(
                expr.left,
                allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
            ),
            "right": _query_expr_canonical(
                expr.right,
                allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
            ),
        }
    if isinstance(expr, BoundUnary):
        if expr.op not in _QUERY_TREE_UNARY_OPERATORS:
            raise ValueError(CADENCE_PREDICATE_QUERY_EXPRESSION_UNSUPPORTED)
        return {
            "kind": "unary", "op": expr.op,
            "operand": _query_expr_canonical(
                expr.operand,
                allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
            ),
        }
    if isinstance(expr, BoundBetween):
        return {
            "kind": "between",
            "operand": _query_expr_canonical(
                expr.operand,
                allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
            ),
            "lower": _query_expr_canonical(
                expr.lower,
                allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
            ),
            "upper": _query_expr_canonical(
                expr.upper,
                allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
            ),
            "negated": expr.negated,
        }
    if isinstance(expr, BoundFunc):
        arity_source = (
            _QUERY_TREE_EXTENDED_FUNCTION_ARITY
            if allow_extended_expression_canonicalization
            else _QUERY_TREE_FUNCTION_ARITY
        )
        arity = arity_source.get(expr.name)
        if arity is None or not arity[0] <= len(expr.args) <= arity[1]:
            raise ValueError(CADENCE_PREDICATE_QUERY_EXPRESSION_UNSUPPORTED)
        if (
            expr.name in {"julianday", "substring", "and", "or", "coalesce", "replace"}
            and expr.distinct
            or expr.name in {"sum", "avg", "min", "max"}
            and (isinstance(expr.args[0], BoundStar))
            or expr.name == "count" and expr.distinct
            and isinstance(expr.args[0], BoundStar)
        ):
            raise ValueError(CADENCE_PREDICATE_QUERY_EXPRESSION_UNSUPPORTED)
        return {
            "kind": "func", "name": expr.name,
            "args": [
                _query_expr_canonical(
                    arg,
                    allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
                )
                for arg in expr.args
            ],
            "distinct": expr.distinct,
        }
    if isinstance(expr, BoundIsNull):
        return {
            "kind": "is_null",
            "operand": _query_expr_canonical(
                expr.operand,
                allow_extended_expression_canonicalization=allow_extended_expression_canonicalization,
            ),
            "negated": expr.negated,
        }
    if isinstance(expr, BoundUnsupported):
        raise ValueError(expr.construct)
    result = _bound_to_canonical(expr)
    if not isinstance(result, dict):
        raise TypeError(f"表达式规范形态不是 object: {type(expr).__name__}")
    return result


def _contains_nested_query_expression(expr: BoundExpr) -> bool:
    """判断表达式是否含需跨 scope 递归保存的查询节点。"""
    return any(
        isinstance(node, (BoundSubquery, BoundScalarSubquery))
        or (isinstance(node, BoundIn) and node.subquery is not None)
        for node in walk_bound_expr(expr)
    )


def _requires_compositional_query_tree(query: BoundQuery) -> bool:
    """仅在集合分支、derived table 或嵌套查询时切到闭合 query-tree 字节形态。"""
    if (
        query.compound is not None
        or isinstance(query.from_table, BoundDerivedTable)
        or any(isinstance(join.table, BoundDerivedTable) for join in query.joins)
    ):
        return True
    expressions: list[BoundExpr] = [
        projection.expr for projection in query.projections
    ]
    expressions.extend(query.group_by)
    if query.where is not None:
        expressions.append(query.where)
    if query.having is not None:
        expressions.append(query.having)
    expressions.extend(item.expr for item in query.order_by)
    expressions.extend(join.on for join in query.joins if join.on is not None)
    return any(_contains_nested_query_expression(expr) for expr in expressions)


def canonical_sql_bytes(bound: BoundQuery) -> bytes:
    """BoundQuery 的规范字节（供任务 4 与追踪层复用）。

    普通单 SELECT 延续既有 bound-IR 字节，避免 successor 因 query-tree 子域
    拒绝旧版可消费的谓词；集合分支、derived table 和嵌套查询则使用闭合 tree
    记录完整 scope。简单 SELECT 的新 LIMIT/OFFSET 仅在声明时显式加入兼容形态。
    """
    if (
        _supports_query_tree(bound.query_contract)
        and _requires_compositional_query_tree(bound)
    ):
        return canonical_bytes(
            canonical_bound_query_tree(
                bound,
                allow_extended_expression_canonicalization=(
                    _supports_expression_extensions(bound.query_contract)
                ),
            )
        )
    return canonical_bytes(_bound_to_canonical(bound))


def _bound_to_canonical(
    bound: (
        BoundExpr
        | BoundProjection
        | BoundOrderItem
        | BoundQuery
        | BoundSetOperation
        | BoundTable
        | BoundDerivedTable
        | BoundJoin
    ),
) -> Any:
    """BoundQuery 结构 → 规范序列化 值域（供身份计算）。"""
    if isinstance(bound, BoundColumn):
        return {
            "kind": "column",
            "relation_id": bound.relation_id,
            "column_id": bound.column_id,
        }
    if isinstance(bound, BoundOuterColumn):
        return _query_expr_canonical(bound)
    if isinstance(bound, BoundDerivedColumn):
        return {
            "kind": "derived_output_slot",
            "scope_path": bound.scope_path,
            "projection_ordinal": str(bound.projection_ordinal),
        }
    if isinstance(bound, BoundLiteral):
        return {"kind": "literal", "typed_tag": bound.typed_tag, "canonical": bound.canonical}
    if isinstance(bound, BoundStar):
        return {"kind": "star"}
    if isinstance(bound, BoundFunc):
        return {
            "kind": "func",
            "name": bound.name,
            "args": [_bound_to_canonical(a) for a in bound.args],
            "distinct": bound.distinct,
        }
    if isinstance(bound, BoundCast):
        return {
            "kind": "cast",
            "target_type": bound.target_type,
            "operand": _bound_to_canonical(bound.operand),
        }
    if isinstance(bound, BoundWindow):
        frame = bound.frame

        def frame_bound_value(value: BoundWindowFrameBound) -> dict[str, Any]:
            if value.kind == "OFFSET" and value.offset is not None:
                return {
                    "kind": value.kind,
                    "side": value.side,
                    "offset": _bound_to_canonical(value.offset),
                }
            if value.kind == "UNBOUNDED":
                return {"kind": value.kind, "side": value.side}
            if value.kind == "CURRENT_ROW":
                return {"kind": value.kind}
            raise TypeError("invalid bound window frame")

        return {
            "kind": "window",
            "function": _bound_to_canonical(bound.function),
            "partition_by": [_bound_to_canonical(item) for item in bound.partition_by],
            "order_by": [_bound_to_canonical(item) for item in bound.order_by],
            "frame": (
                None
                if frame is None
                else {
                    "unit": frame.unit,
                    "start": frame_bound_value(frame.start),
                    "end": frame_bound_value(frame.end),
                    "exclusion": frame.exclusion,
                }
            ),
        }
    if isinstance(bound, BoundBinary):
        return {
            "kind": "binary",
            "op": bound.op,
            "left": _bound_to_canonical(bound.left),
            "right": _bound_to_canonical(bound.right),
        }
    if isinstance(bound, BoundUnary):
        return {
            "kind": "unary",
            "op": bound.op,
            "operand": _bound_to_canonical(bound.operand),
        }
    if isinstance(bound, BoundBetween):
        return {
            "kind": "between",
            "operand": _bound_to_canonical(bound.operand),
            "lower": _bound_to_canonical(bound.lower),
            "upper": _bound_to_canonical(bound.upper),
            "negated": bound.negated,
        }
    if isinstance(bound, BoundIn):
        # literal 分支不新增 subquery:null，保持既有 canonical 字节形状；
        # 局部来源显式序列化，空 values 本身永远不表示空子查询结果。
        result = {
            "kind": "in",
            "operand": _bound_to_canonical(bound.operand),
            "values": [_bound_to_canonical(v) for v in bound.values],
            "negated": bound.negated,
        }
        if bound.subquery is not None:
            if bound.values:
                raise _fail(CADENCE_PREDICATE_SOURCE_UNBOUND, "membership 来源混用")
            result["subquery"] = canonical_bound_subquery(bound.subquery)
        return result
    if isinstance(bound, BoundSubquery):
        return canonical_bound_subquery(bound)
    if isinstance(bound, BoundScalarSubquery):
        return {"kind": "scalar_subquery", "query": _bound_to_canonical(bound.query)}
    if isinstance(bound, BoundSetOperation):
        return {
            "operator": bound.operator,
            "quantifier": bound.quantifier,
            "left": _bound_to_canonical(bound.left),
            "right": _bound_to_canonical(bound.right),
        }
    if isinstance(bound, BoundNotExistsCorrelated):
        # NOT EXISTS 的 canonical 只保留绑定后的关系/列身份；不保存 SQL、alias
        # 或执行结果，且不能改写成 NOT IN / anti-join，否则会丢失 correlation
        # scope 与 SQL NULL witness 的可审计边界。
        return {
            "kind": "not_exists_correlated",
            "source_table": {"relation_id": bound.source_table.relation_id},
            "inner_column": _bound_to_canonical(bound.inner_column),
            "outer_column": _bound_to_canonical(bound.outer_column),
        }
    if isinstance(bound, BoundIsNull):
        return {
            "kind": "is_null",
            "operand": _bound_to_canonical(bound.operand),
            "negated": bound.negated,
        }
    if isinstance(bound, BoundUnsupported):
        return {"kind": "unsupported", "construct": bound.construct}
    if isinstance(bound, BoundProjection):
        return {"expr": _bound_to_canonical(bound.expr), "alias": bound.alias}
    if isinstance(bound, BoundOrderItem):
        # BoundOrderItem canonical 只含 expr/direction/null_order；
        # 不携带 alias、原始 SQL 文本或 source span（identity 字节纯净性）。
        return {
            "expr": _bound_to_canonical(bound.expr),
            "direction": bound.direction,
            "null_order": bound.null_order,
        }
    if isinstance(bound, BoundTable):
        return {"relation_id": bound.relation_id, "alias": bound.alias}
    if isinstance(bound, BoundDerivedTable):
        return {
            "scope_path": bound.scope_path,
            "alias": bound.alias,
            "query": _bound_to_canonical(bound.query),
            "output_names": list(bound.output_names),
        }
    if isinstance(bound, BoundJoin):
        return {
            "table": _bound_to_canonical(bound.table),
            "join_type": bound.join_type,
            "on": _bound_to_canonical(bound.on) if bound.on is not None else None,
        }
    if isinstance(bound, BoundQuery):
        result = {
            "projections": [_bound_to_canonical(p) for p in bound.projections],
            "from_table": (
                _bound_to_canonical(bound.from_table) if bound.from_table is not None else None
            ),
            "joins": [_bound_to_canonical(j) for j in bound.joins],
            "where": _bound_to_canonical(bound.where) if bound.where is not None else None,
            "group_by": [_bound_to_canonical(e) for e in bound.group_by],
            "order_by": [_bound_to_canonical(e) for e in bound.order_by],
            "ctes": [_bound_to_canonical(c) for c in bound.ctes],
            "distinct": bound.distinct,
        }
        if bound.having is not None:
            result["having"] = _bound_to_canonical(bound.having)
        if bound.compound is not None:
            result["compound"] = _bound_to_canonical(bound.compound)
        if bound.limit is not None:
            result["limit"] = str(bound.limit)
        if bound.offset is not None:
            result["offset"] = str(bound.offset)
        return result
    raise TypeError(f"无法投影的 Bound 节点: {type(bound).__name__}")


# ---------------------------------------------------------------------------
# 只读遍历与列收集
# ---------------------------------------------------------------------------
def walk_bound_expr(expr: BoundExpr) -> Iterator[BoundExpr]:
    """只读深度优先遍历 BoundExpr 树，产出含根在内的所有节点。

    覆盖 column / literal / star / func / binary / unary / between / in /
    is_null / unsupported。局部 subquery 保持 opaque，禁止把内层列归到外层
    SOURCE_RELATION；其身份由 successor canonical 单独保存。不遍历 BoundQuery、
    BoundProjection、BoundJoin、BoundOrderItem 等外层结构；调用方需自行从
    projections[*].expr、where、
    joins[*].on、group_by[*]、order_by[*].expr、ctes[*] 等入口进入。

    本函数只读遍历，不修改节点、不生成 SQL、不做类型推断；表达式检查与列收集共用此入口。
    """
    yield expr
    if isinstance(expr, BoundFunc):
        for arg in expr.args:
            yield from walk_bound_expr(arg)
    elif isinstance(expr, BoundCast):
        yield from walk_bound_expr(expr.operand)
    elif isinstance(expr, BoundWindow):
        yield from walk_bound_expr(expr.function)
        for item in expr.partition_by:
            yield from walk_bound_expr(item)
        for order_item in expr.order_by:
            yield from walk_bound_expr(order_item.expr)
        if expr.frame is not None:
            for boundary in (expr.frame.start, expr.frame.end):
                if boundary.offset is not None:
                    yield from walk_bound_expr(boundary.offset)
    elif isinstance(expr, BoundBinary):
        yield from walk_bound_expr(expr.left)
        yield from walk_bound_expr(expr.right)
    elif isinstance(expr, BoundUnary):
        yield from walk_bound_expr(expr.operand)
    elif isinstance(expr, BoundBetween):
        yield from walk_bound_expr(expr.operand)
        yield from walk_bound_expr(expr.lower)
        yield from walk_bound_expr(expr.upper)
    elif isinstance(expr, BoundIn):
        yield from walk_bound_expr(expr.operand)
        for value in expr.values:
            yield from walk_bound_expr(value)
    elif isinstance(expr, BoundIsNull):
        yield from walk_bound_expr(expr.operand)
    # BoundColumn / BoundLiteral / BoundStar / BoundUnsupported 无子女。


def collect_bound_columns(expr: BoundExpr) -> set[tuple[str, str]]:
    """收集表达式树中所有 BoundColumn 的 (relation_id, column_id) 集合。

    用于 JOIN ON / WHERE / ORDER BY / GROUP BY 等任意表达式入口的列归属分析。
    与 ``walk_bound_expr`` 配合，保证列收集不漏任何子表达式。
    """
    return {
        (node.relation_id, node.column_id)
        for node in walk_bound_expr(expr)
        if isinstance(node, BoundColumn)
    }
