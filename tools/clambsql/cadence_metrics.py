"""CLAMBSQL 上 CADENCE 的 EX 与两种歧义评分口径。

此模块只读取终态预测、发布标注和 SQLite 数据库，不参与候选分组、probe 选择、
Memory 决策或回答。归因短语分数使用归一化精确匹配，仅作为本地诊断口径。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import sqlglot
from sqlglot import exp

from tools.clambsql.sqlite_execution import _result_signature

CADENCE_METRIC_INDEX_MISMATCH = "CADENCE_METRIC_INDEX_MISMATCH"
CADENCE_METRIC_INPUT_INVALID = "CADENCE_METRIC_INPUT_INVALID"
CADENCE_METRIC_ANNOTATION_INVALID = "CADENCE_METRIC_ANNOTATION_INVALID"


def read_prediction_jsonl(path: Path) -> dict[int, Mapping[str, Any]]:
    """读取闭合的逐请求预测 JSONL，并拒绝重复或缺少 request index 的行。"""

    outputs: dict[int, Mapping[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(CADENCE_METRIC_INPUT_INVALID) from exc
    for line in lines:
        if not line.strip():
            continue
        try:
            output = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(CADENCE_METRIC_INPUT_INVALID) from exc
        if not isinstance(output, Mapping):
            raise ValueError(CADENCE_METRIC_INPUT_INVALID)
        index = output.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index in outputs:
            raise ValueError(CADENCE_METRIC_INDEX_MISMATCH)
        if "final_sql" not in output or "predicted_ambiguities" not in output:
            raise ValueError(CADENCE_METRIC_INPUT_INVALID)
        outputs[index] = output
    return outputs


def _normalized_phrase(value: str) -> str:
    """将短语做 Unicode NFKC、大小写折叠和词边界规范化后用于精确比较。"""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _annotation_phrases(record: Mapping[str, Any]) -> frozenset[str]:
    """提取 CLAMBSQL match/query 标注的短语键，并合并同一短语的重复标注。"""

    ambiguity = record.get("gold_ambiguity")
    if not isinstance(ambiguity, Mapping):
        raise ValueError(CADENCE_METRIC_ANNOTATION_INVALID)
    phrases: set[str] = set()
    for label in ("match", "query"):
        raw = ambiguity.get(label)
        if isinstance(raw, str):
            try:
                annotations = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(CADENCE_METRIC_ANNOTATION_INVALID) from exc
        else:
            annotations = raw
        if not isinstance(annotations, Mapping):
            raise ValueError(CADENCE_METRIC_ANNOTATION_INVALID)
        for phrase in annotations:
            if not isinstance(phrase, str) or not phrase.strip():
                raise ValueError(CADENCE_METRIC_ANNOTATION_INVALID)
            normalized = _normalized_phrase(phrase)
            if normalized:
                phrases.add(normalized)
    return frozenset(phrases)


def _predicted_phrases(output: Mapping[str, Any]) -> frozenset[str]:
    """规范化预测归因短语；重复识别只计一次，空值不会被当成标签。"""

    values = output.get("predicted_ambiguities")
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(CADENCE_METRIC_INPUT_INVALID)
    phrases: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError(CADENCE_METRIC_INPUT_INVALID)
        normalized = _normalized_phrase(value)
        if normalized:
            phrases.add(normalized)
    return frozenset(phrases)


def _schema_entities(sql: str) -> frozenset[tuple[str, str]]:
    """抽取 SQL 中的表名和列名实体；别名不计入，实体类型与名称共同构成 ID。"""

    tree = sqlglot.parse_one(sql, read="sqlite")
    entities: set[tuple[str, str]] = set()
    for table in tree.find_all(exp.Table):
        if table.name:
            entities.add(("table", table.name.casefold()))
    for column in tree.find_all(exp.Column):
        if column.name and column.name != "*":
            entities.add(("column", column.name.casefold()))
    return frozenset(entities)


def _set_counts(
    gold: frozenset[Any], predicted: frozenset[Any]
) -> dict[str, int | float]:
    """以集合交并差计算逐请求 P/R/F1 和 AllFound 的充分统计量。"""

    true_positive = len(gold & predicted)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "all_found": float(gold.issubset(predicted)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int | float]]:
    """同时返回按请求宏平均和全局计数微平均；AllFound 均按请求求比例。"""

    count = len(rows)
    if count == 0:
        raise ValueError(CADENCE_METRIC_INPUT_INVALID)
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in ("true_positive", "false_positive", "false_negative")
    }
    micro_precision_denominator = totals["true_positive"] + totals["false_positive"]
    micro_recall_denominator = totals["true_positive"] + totals["false_negative"]
    micro_precision = (
        totals["true_positive"] / micro_precision_denominator
        if micro_precision_denominator
        else 0.0
    )
    micro_recall = (
        totals["true_positive"] / micro_recall_denominator
        if micro_recall_denominator
        else 0.0
    )
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    metric_names = ("precision", "recall", "f1", "all_found")
    return {
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "all_found": fmean(float(row["all_found"]) for row in rows),
            **totals,
            "request_count": count,
        },
        "macro": {
            **{
                name: fmean(float(row[name]) for row in rows)
                for name in metric_names
            },
            "request_count": count,
        },
    }


def build_cadence_metrics(
    records: Sequence[Mapping[str, Any]],
    outputs_by_index: Mapping[int, Mapping[str, Any]],
    *,
    database_root: Path,
) -> dict[str, Any]:
    """对 CLAMBSQL 最终 SQL 与两种歧义集计算指标，缺失 SQL 作为失败留在分母。"""

    indices = [record.get("index") for record in records]
    if (
        not records
        or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or len(indices) != len(set(indices))
        or set(indices) != set(outputs_by_index)
    ):
        raise ValueError(CADENCE_METRIC_INDEX_MISMATCH)

    schema_rows: list[dict[str, int | float]] = []
    annotation_rows: list[dict[str, int | float]] = []
    per_request_rows: list[dict[str, Any]] = []
    execution_matches = 0
    execution_errors = 0
    pool_member_count = 0
    off_pool_count = 0
    schema_sql_parse_failures = 0
    for record in records:
        index = record["index"]
        output = outputs_by_index[index]
        if not isinstance(output, Mapping):
            raise ValueError(CADENCE_METRIC_INPUT_INVALID)
        gold_sql = record.get("gold_query")
        db_file = record.get("db_file")
        candidate_sqls = record.get("ambiguous_queries")
        final_sql = output.get("final_sql")
        if (
            not isinstance(gold_sql, str)
            or not gold_sql.strip()
            or not isinstance(db_file, str)
            or not db_file.strip()
            or isinstance(candidate_sqls, (str, bytes))
            or not isinstance(candidate_sqls, Sequence)
            or any(not isinstance(candidate, str) for candidate in candidate_sqls)
            or (final_sql is not None and not isinstance(final_sql, str))
        ):
            raise ValueError(CADENCE_METRIC_INPUT_INVALID)
        if final_sql is not None and final_sql.strip():
            if final_sql in candidate_sqls:
                pool_member_count += 1
            else:
                off_pool_count += 1

        gold_schema = _schema_entities(gold_sql)
        if final_sql is None or not final_sql.strip():
            predicted_schema: frozenset[tuple[str, str]] = frozenset()
        else:
            try:
                predicted_schema = _schema_entities(final_sql)
            except Exception:  # noqa: BLE001 - invalid final SQL is an empty prediction.
                predicted_schema = frozenset()
                schema_sql_parse_failures += 1
        schema_scores = _set_counts(gold_schema, predicted_schema)
        annotation_scores = _set_counts(
            _annotation_phrases(record), _predicted_phrases(output)
        )
        schema_rows.append(schema_scores)
        annotation_rows.append(annotation_scores)

        execution_match = False
        if final_sql is not None and final_sql.strip():
            database = (database_root / db_file).resolve()
            try:
                execution_match = (
                    _result_signature(database, final_sql)
                    == _result_signature(database, gold_sql)
                )
                execution_matches += int(execution_match)
            except Exception:  # noqa: BLE001 - execution failures count as misses.
                execution_errors += 1
        per_request_rows.append(
            {
                "index": index,
                "final_sql_ready": final_sql is not None and bool(final_sql.strip()),
                "final_sql_pool_member": final_sql in candidate_sqls,
                "execution_match": execution_match,
                "schema_entity": schema_scores,
                "annotation_phrase": annotation_scores,
            }
        )

    denominator = len(records)
    return {
        "schema_version": "CadenceCLAMBSQLMetrics",
        "request_count": denominator,
        "per_request": per_request_rows,
        "denominator_rule": "all requests; missing/invalid final SQL counts as a miss",
        "execution_match": {
            "definition": "typed SQLite result multiset equality against gold_query",
            "count": execution_matches,
            "denominator": denominator,
            "rate": execution_matches / denominator,
            "execution_error_count": execution_errors,
        },
        "final_sql_pool_membership": {
            "member_count": pool_member_count,
            "denominator": denominator,
            "rate": pool_member_count / denominator,
            "off_pool_count": off_pool_count,
        },
        "schema_entity_metrics": {
            "scorer": (
                "local SQLGlot table/column-name set scorer; "
                "legacy scorer source unavailable"
            ),
            "all_found": "gold entity set is a subset of predicted entity set",
            "sql_parse_failure_count": schema_sql_parse_failures,
            **_aggregate(schema_rows),
        },
        "annotation_phrase_metrics": {
            "scorer": (
                "normalized exact matching of attribution phrases to "
                "gold_ambiguity match/query keys"
            ),
            "normalization": (
                "Unicode NFKC, casefold, punctuation-to-boundary tokenization, "
                "whitespace collapse"
            ),
            "duplicate_policy": "deduplicate normalized phrases per request",
            "all_found": "all distinct normalized gold phrase keys were predicted",
            **_aggregate(annotation_rows),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """从 CLAMBSQL JSON 与逐请求预测文件写出可复查的指标 JSON。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        dataset_bytes = args.dataset.read_bytes()
        prediction_bytes = args.predictions.read_bytes()
        records = json.loads(dataset_bytes.decode("utf-8"))
        if not isinstance(records, list) or not all(
            isinstance(record, Mapping) for record in records
        ):
            raise ValueError(CADENCE_METRIC_INPUT_INVALID)
        outputs = read_prediction_jsonl(args.predictions)
        metrics = build_cadence_metrics(
            records,
            outputs,
            database_root=args.database_root,
        )
        metrics["input_hashes"] = {
            "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

