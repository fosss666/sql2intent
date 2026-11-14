"""SQLGlot AST 到 CADENCE typed expression representation 的局部映射。

本模块只识别可无损保存的 AST 子树，并把 SQLGlot 节点和参数顺序交给既有
`BoundExpr` binder；不生成或改写 SQL 字符串。超出本地闭合形态时返回稳定的
unsupported 结果，避免漏掉聚合修饰符后错误合并候选身份。
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp


@dataclass(frozen=True)
class SqlglotFunctionMapping:
    """SQLite 聚合函数的 AST 映射结果；reason code 非空时不得消费 arguments。"""

    arguments: tuple[exp.Expression, ...]
    distinct: bool
    reason_code: str | None = None
    detail: str = ""

    @property
    def supported(self) -> bool:
        """表示 AST 是否完整落入已声明的 mapping 子域。"""
        return self.reason_code is None


def map_sqlite_group_concat_ast(
    node: exp.Expression,
) -> SqlglotFunctionMapping | None:
    """映射 `GROUP_CONCAT(X)` / `GROUP_CONCAT(X, Y)` 的 SQLite AST。

    保留表达式 X、可选 separator Y、DISTINCT 和参数顺序。聚合内部 ORDER BY、
    空参数、额外参数字段及多表达式 DISTINCT 目前没有对应 typed node，必须
    返回明确拒绝，不能通过通用参数遍历把它们压扁或忽略。
    """
    if not isinstance(node, exp.GroupConcat):
        return None

    unexpected_args = {
        key
        for key, value in node.args.items()
        if value is not None and key not in {"this", "separator"}
    }
    if unexpected_args:
        return SqlglotFunctionMapping(
            (),
            False,
            "EXPR_GROUP_CONCAT_SHAPE_UNSUPPORTED",
            "GROUP_CONCAT 含未建模的 SQLGlot 参数字段",
        )

    raw_value = node.args.get("this")
    if isinstance(raw_value, exp.Order):
        return SqlglotFunctionMapping(
            (),
            False,
            "EXPR_GROUP_CONCAT_ORDER_UNSUPPORTED",
            "GROUP_CONCAT 内部 ORDER BY 尚无 typed query IR 节点",
        )
    if not isinstance(raw_value, exp.Expression):
        return SqlglotFunctionMapping(
            (),
            False,
            "EXPR_GROUP_CONCAT_SHAPE_UNSUPPORTED",
            "GROUP_CONCAT 缺少值表达式",
        )

    distinct = isinstance(raw_value, exp.Distinct)
    if distinct:
        distinct_values = list(raw_value.expressions)
        if len(distinct_values) != 1 or not isinstance(
            distinct_values[0], exp.Expression
        ):
            return SqlglotFunctionMapping(
                (),
                False,
                "EXPR_GROUP_CONCAT_SHAPE_UNSUPPORTED",
                "GROUP_CONCAT DISTINCT 必须且只能包住一个表达式",
            )
        value_expression = distinct_values[0]
    else:
        value_expression = raw_value

    arguments = [value_expression]
    separator = node.args.get("separator")
    if separator is not None:
        if not isinstance(separator, exp.Expression):
            return SqlglotFunctionMapping(
                (),
                False,
                "EXPR_GROUP_CONCAT_SHAPE_UNSUPPORTED",
                "GROUP_CONCAT separator 不是 SQLGlot 表达式",
            )
        arguments.append(separator)
    return SqlglotFunctionMapping(tuple(arguments), distinct)
