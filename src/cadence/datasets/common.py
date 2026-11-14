"""候选池标准请求结构与文件读取工具。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 封闭 reason code（数据集适配器层；与 EVAL_* / ENV_* 命名空间隔离）。
# ---------------------------------------------------------------------------
DATASET_FILE_NOT_FOUND = "DATASET_FILE_NOT_FOUND"
DATASET_JSON_INVALID = "DATASET_JSON_INVALID"
DATASET_MISSING_REQUIRED_FIELD = "DATASET_MISSING_REQUIRED_FIELD"
DATASET_UNSUPPORTED_FORMAT = "DATASET_UNSUPPORTED_FORMAT"
DATASET_TOO_FEW_CANDIDATES = "DATASET_TOO_FEW_CANDIDATES"


class DatasetAdapterRejected(Exception):
    """数据集适配器 fail-closed。``reason_code`` 为 DATASET_* 封闭码。"""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(reason_code, detail)
        self.reason_code = reason_code
        self.detail = detail


def _fail(code: str, detail: str) -> DatasetAdapterRejected:
    return DatasetAdapterRejected(code, detail)


def _read_json(path: Path) -> Any:
    """fail-closed 读取 JSON 文件；文件不存在/JSON 损坏抛稳定码。"""
    if not path.exists() or not path.is_file():
        raise _fail(DATASET_FILE_NOT_FOUND, f"数据文件不存在: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise _fail(DATASET_JSON_INVALID, f"JSON 解析失败 {path}: {exc}") from exc


def _required_text(row: Mapping[str, Any], field: str, line_number: int) -> str:
    """读取统一请求中的必填文本字段。"""

    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise _fail(
            DATASET_MISSING_REQUIRED_FIELD,
            f"候选池第 {line_number} 行缺少有效字段 {field}",
        )
    return value


def sha256_of_file(path: Path) -> str:
    """文件字节 SHA-256（DatasetManifest.file_sha256 用）。"""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class StandardRequest:
    """统一候选池输入；Gold SQL 仅供显式启用的模拟用户使用。

    属性：
    - request_id：候选池内唯一的请求标识；
    - question_text：自然语言问题原文；
    - catalog：SchemaCatalog 定义（dialect_id + relations）；
    - database_path：数据库文件路径，供执行器使用；
    - candidates：含 candidate_id 和 sql_text 的候选序列；
    - locale：供模型回调使用的语言；
    - engine_name：SQL 执行后端，可由调用方显式提供。
    - temporal_columns：由来源 Schema 显式提供的时间列身份；不从列名猜测。
    - gold_sql：可选参考 SQL；因子化、归因和问句提示都不会读取它。
    """

    request_id: str
    question_text: str
    catalog: dict[str, Any]
    database_path: str
    candidates: tuple[dict[str, Any], ...]
    locale: str
    engine_name: str | None = None
    temporal_columns: tuple[tuple[str, str], ...] = ()
    gold_sql: str | None = None


def load_candidate_pool_jsonl(path: Path) -> tuple[StandardRequest, ...]:
    """读取统一 JSONL 候选池；可选 temporal_columns 显式标记时间列。"""

    if not path.is_file():
        raise _fail(DATASET_FILE_NOT_FOUND, f"候选池文件不存在: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise _fail(DATASET_FILE_NOT_FOUND, f"候选池文件不可读: {path}") from exc

    requests: list[StandardRequest] = []
    seen_request_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _fail(
                DATASET_JSON_INVALID, f"候选池第 {line_number} 行不是有效 JSON"
            ) from exc
        if not isinstance(row, dict):
            raise _fail(
                DATASET_UNSUPPORTED_FORMAT,
                f"候选池第 {line_number} 行必须是 JSON object",
            )

        request_id = _required_text(row, "request_id", line_number)
        if request_id in seen_request_ids:
            raise _fail(
                DATASET_UNSUPPORTED_FORMAT,
                f"候选池 request_id 重复: {request_id}",
            )
        seen_request_ids.add(request_id)

        catalog = row.get("catalog")
        candidates = row.get("candidates")
        if not isinstance(catalog, dict):
            raise _fail(
                DATASET_MISSING_REQUIRED_FIELD,
                f"候选池第 {line_number} 行 catalog 必须是 object",
            )
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise _fail(
                DATASET_TOO_FEW_CANDIDATES,
                f"候选池第 {line_number} 行至少需要两个候选 SQL",
            )
        normalized_candidates: list[dict[str, Any]] = []
        candidate_ids: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise _fail(
                    DATASET_UNSUPPORTED_FORMAT,
                    f"候选池第 {line_number} 行候选项必须是 object",
                )
            candidate_id = candidate.get("candidate_id")
            sql_text = candidate.get("sql_text")
            if (
                not isinstance(candidate_id, str)
                or not candidate_id.strip()
                or not isinstance(sql_text, str)
                or not sql_text.strip()
            ):
                raise _fail(
                    DATASET_MISSING_REQUIRED_FIELD,
                    f"候选池第 {line_number} 行候选项需要 candidate_id 和 sql_text",
                )
            if candidate_id in candidate_ids:
                raise _fail(
                    DATASET_UNSUPPORTED_FORMAT,
                    f"候选池第 {line_number} 行 candidate_id 重复: {candidate_id}",
                )
            candidate_ids.add(candidate_id)
            normalized_candidates.append(
                {"candidate_id": candidate_id, "sql_text": sql_text}
            )

        engine_name = row.get("engine_name")
        if engine_name is not None and (
            not isinstance(engine_name, str) or not engine_name.strip()
        ):
            raise _fail(
                DATASET_UNSUPPORTED_FORMAT,
                f"候选池第 {line_number} 行 engine_name 无效",
            )
        gold_sql = row.get("gold_sql")
        if gold_sql is not None and (
            not isinstance(gold_sql, str) or not gold_sql.strip()
        ):
            raise _fail(
                DATASET_UNSUPPORTED_FORMAT,
                f"候选池第 {line_number} 行 gold_sql 无效",
            )
        temporal_columns = _temporal_columns(row.get("temporal_columns", []), line_number)
        requests.append(
            StandardRequest(
                request_id=request_id,
                question_text=_required_text(row, "question_text", line_number),
                catalog=catalog,
                database_path=_required_text(row, "database_path", line_number),
                candidates=tuple(normalized_candidates),
                locale=_required_text(row, "locale", line_number),
                engine_name=engine_name,
                temporal_columns=temporal_columns,
                gold_sql=gold_sql,
            )
        )
    if not requests:
        raise _fail(DATASET_UNSUPPORTED_FORMAT, "候选池没有请求记录")
    return tuple(requests)


def _temporal_columns(value: Any, line_number: int) -> tuple[tuple[str, str], ...]:
    """读取候选池提供的 relation/column 对，不通过名字推断时间属性。"""

    if not isinstance(value, list):
        raise _fail(
            DATASET_UNSUPPORTED_FORMAT,
            f"候选池第 {line_number} 行 temporal_columns 必须是数组",
        )
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"relation_id", "column_id"}
            or not isinstance(item.get("relation_id"), str)
            or not item["relation_id"].strip()
            or not isinstance(item.get("column_id"), str)
            or not item["column_id"].strip()
        ):
            raise _fail(
                DATASET_UNSUPPORTED_FORMAT,
                f"候选池第 {line_number} 行 temporal_columns 项必须含 relation_id 和 column_id",
            )
        pair = (item["relation_id"], item["column_id"])
        if pair in seen:
            raise _fail(
                DATASET_UNSUPPORTED_FORMAT,
                f"候选池第 {line_number} 行 temporal_columns 存在重复项",
            )
        result.append(pair)
        seen.add(pair)
    return tuple(result)


@dataclass(frozen=True)
class DatasetManifest:
    """数据集来源溯源清单（每 cohort 一份）。

    所有外部候选池进入 CADENCE 前记录来源、许可和文件哈希，
    数据库快照位置、split、适配器版本。缺任一字段即视为不可复现。

    snapshot_profile
    ----------------
    声明本 cohort 使用的数据库快照剖面（lineage 标记，不做路径猜测）：
    - ``"legacy"``：未显式指定剖面的旧快照（如 dev all-VARCHAR 冒烟快照），
      旧 smoke 调用不传本字段时的默认值，保持向后兼容；
    - ``"evaluation_typed"``：保留 SQLite 声明类型的 evaluation-grade DuckDB
      快照（``data/processed/spider_dev_evaluation``）；
    - ``"native_sqlite"``：直接使用原始 SQLite 数据文件，只读执行。
    适配器**不**根据该字段反推 ``db_snapshot_root``；调用方必须同时显式传入
    与剖面一致的完整 root。
    """

    source: str
    license: str
    version: str
    file_sha256: str
    database_snapshot: str
    split: str
    adapter_version: str
    snapshot_profile: str = "legacy"
    engine_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "source": self.source,
            "license": self.license,
            "version": self.version,
            "file_sha256": self.file_sha256,
            "database_snapshot": self.database_snapshot,
            "split": self.split,
            "adapter_version": self.adapter_version,
            "snapshot_profile": self.snapshot_profile,
        }
        if self.engine_name is not None:
            out["engine_name"] = self.engine_name
        return out


