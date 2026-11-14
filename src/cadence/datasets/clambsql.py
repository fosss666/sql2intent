"""读取 CLAMBSQL 发布包并保留原始候选 SQL。

候选 SQL 不会被生成、筛选或改写。数据库按发布的 ``db_file`` 隔离，因为 ``db_id``
不是唯一键。
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from cadence.datasets.common import (
    DATASET_FILE_NOT_FOUND,
    DATASET_MISSING_REQUIRED_FIELD,
    DATASET_TOO_FEW_CANDIDATES,
    DATASET_UNSUPPORTED_FORMAT,
    DatasetManifest,
    StandardRequest,
    _fail,
    _read_json,
    sha256_of_file,
)

ADAPTER_VERSION = "clambsql-adapter-1.2.0"
_CLAMBSQL_LOCALE = "en-US"
_TEMPORAL_DECLARED_TYPES = frozenset(
    {"DATE", "DATETIME", "DATETIME2", "TIME", "TIME2", "TIMESTAMP", "YEAR"}
)


@dataclass(frozen=True)
class ClambsqlRecord:
    """一条发布记录及其单独的运行请求。

    发布标注和 schema 字符串保留原样，仅供 Gold-oracle 评测和离线计分；
    方法调用只接收 ``request``，不会收到这些字段。
    """

    request: StandardRequest
    index: int
    db_id: str
    ambig_type: str
    db_file: str
    schema_without_content: str
    schema_with_content: str
    gold_ambiguity: dict[str, Any]
    clarification_context: str
    clear_ambiguity: str
    gold_query: str


@dataclass(frozen=True)
class ClambsqlDataset:
    """CLAMBSQL 记录、运行请求和通用来源清单的只读装配结果。"""

    records: tuple[ClambsqlRecord, ...]
    manifest: DatasetManifest

    @property
    def requests(self) -> tuple[StandardRequest, ...]:
        """返回不含 Gold/标注字段的标准运行请求。"""
        return tuple(record.request for record in self.records)


def _required_string(record: Mapping[str, Any], field: str, row: int) -> str:
    """验证必需字符串字段；缺失与空值均 fail-closed，避免静默造默认值。"""
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise _fail(
            DATASET_MISSING_REQUIRED_FIELD,
            f"CLAMBSQL 第 {row} 行缺失或无效字段 {field}",
        )
    return value


def _validated_db_file(value: str, row: int) -> PurePosixPath:
    """校验发布包的 POSIX 数据库相对路径，拒绝绝对路径和目录逃逸。"""
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or PureWindowsPath(value).drive
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
        or path.suffix.lower() != ".sqlite"
    ):
        raise _fail(
            DATASET_UNSUPPORTED_FORMAT,
            f"CLAMBSQL 第 {row} 行 db_file 不是包内相对 POSIX .sqlite 路径: {value!r}",
        )
    return path


def _validate_snapshot_root(value: str) -> PurePosixPath:
    """要求快照根为仓库内 POSIX 路径，保证生成的 EvaluationInput 可移植。"""
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or PureWindowsPath(value).drive
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise _fail(
            DATASET_UNSUPPORTED_FORMAT,
            f"db_snapshot_root 必须是仓库内相对 POSIX 路径: {value!r}",
        )
    return path


def _sqlite_catalog(
    sqlite_path: Path, *, db_file: str, dialect_id: str
) -> tuple[dict[str, Any], tuple[tuple[str, str], ...]]:
    """从原始 SQLite 元数据构建 catalog，不解析有损的展示型 schema 字符串。

    表名按 Unicode 字符串升序，列按 SQLite 声明顺序。只把声明类型明确为
    DATE/TIME/YEAR 的列标记为时间属性，不按列名推断。
    """
    uri = f"{sqlite_path.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        try:
            table_rows = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            relations: list[dict[str, Any]] = []
            temporal_columns: list[tuple[str, str]] = []
            for (table_name,) in table_rows:
                if not isinstance(table_name, str) or not table_name:
                    raise _fail(
                        DATASET_UNSUPPORTED_FORMAT,
                        f"SQLite 表名无效: {db_file}",
                    )
                quoted_name = table_name.replace('"', '""')
                column_rows = connection.execute(
                    f'PRAGMA table_info("{quoted_name}")'
                ).fetchall()
                columns = [str(column[1]) for column in column_rows]
                if not columns or any(not column for column in columns):
                    raise _fail(
                        DATASET_UNSUPPORTED_FORMAT,
                        f"SQLite 表缺少有效列: {db_file}:{table_name}",
                    )
                relations.append({"relation_id": table_name, "columns": columns})
                for column in column_rows:
                    column_id = str(column[1])
                    declared_type = str(column[2] or "")
                    type_tokens = set(re.findall(r"[A-Z]+", declared_type.upper()))
                    if type_tokens.intersection(_TEMPORAL_DECLARED_TYPES):
                        temporal_columns.append((table_name, column_id))
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise _fail(
            DATASET_UNSUPPORTED_FORMAT,
            f"CLAMBSQL SQLite schema 读取失败 {db_file}: {exc}",
        ) from exc
    if not relations:
        raise _fail(
            DATASET_UNSUPPORTED_FORMAT,
            f"CLAMBSQL SQLite 数据库没有可用用户表: {db_file}",
        )
    return (
        {"dialect_id": dialect_id, "relations": relations},
        tuple(temporal_columns),
    )


def load_clambsql_dataset(
    *,
    data_path: Path,
    source_database_root: Path,
    db_snapshot_root: str,
    revision: str,
    license: str = "UNKNOWN",
    engine_name: str = "DUCKDB",
) -> ClambsqlDataset:
    """读取固定版本 CLAMBSQL 原始 JSON，并映射成标准多候选请求。

    ``source_database_root`` 指向发布 ZIP 解包后的 ``database`` 目录。DUCKDB
    模式下 ``db_snapshot_root`` 指向相同布局的 DuckDB 快照根；SQLITE 模式下
    它必须指向可从仓库根访问的原始 SQLite 文件目录。适配器只生成路径，不转换数据库。

    ``revision`` 必须由调用方显式传入。当前数据许可仍未由 G01 确认，默认
    ``license=UNKNOWN`` 只表达未核实状态，不构成可用性或再分发许可判断。
    """
    raw = _read_json(data_path)
    if not isinstance(raw, list):
        raise _fail(DATASET_UNSUPPORTED_FORMAT, "CLAMBSQL 顶层必须是 JSON 数组")
    if not revision.strip():
        raise _fail(DATASET_MISSING_REQUIRED_FIELD, "CLAMBSQL revision 不能为空")
    # 适配器只声明当前已实现的执行后端；SQLite 文件绝不因后缀而自动选路。
    if engine_name not in {"DUCKDB", "SQLITE"}:
        raise _fail(
            DATASET_UNSUPPORTED_FORMAT,
            f"CLAMBSQL adapter 不支持显式 engine_name={engine_name!r}",
        )
    snapshot_root = _validate_snapshot_root(db_snapshot_root)
    source_root = source_database_root.resolve()

    records: list[ClambsqlRecord] = []
    seen_indices: set[int] = set()
    catalog_cache: dict[
        str, tuple[dict[str, Any], tuple[tuple[str, str], ...]]
    ] = {}
    for row, item in enumerate(raw):
        if not isinstance(item, dict):
            raise _fail(DATASET_UNSUPPORTED_FORMAT, f"CLAMBSQL 第 {row} 行不是 object")

        index_value = item.get("index")
        if (
            not isinstance(index_value, int)
            or isinstance(index_value, bool)
            or index_value < 0
        ):
            raise _fail(
                DATASET_MISSING_REQUIRED_FIELD,
                f"CLAMBSQL 第 {row} 行缺少非负整数 index",
            )
        if index_value in seen_indices:
            raise _fail(
                DATASET_UNSUPPORTED_FORMAT,
                f"CLAMBSQL index 重复: {index_value}",
            )
        seen_indices.add(index_value)

        db_id = _required_string(item, "db_id", row)
        ambig_type = _required_string(item, "ambig_type", row)
        db_file_value = _required_string(item, "db_file", row)
        db_file = _validated_db_file(db_file_value, row)
        question = _required_string(item, "question", row)
        schema_without_content = _required_string(item, "schema_without_content", row)
        schema_with_content = _required_string(item, "schema_with_content", row)
        clarification_context = _required_string(item, "clarification_context", row)
        clear_ambiguity = _required_string(item, "clear_ambiguity", row)
        gold_query = _required_string(item, "gold_query", row)

        candidates_value = item.get("ambiguous_queries")
        if not isinstance(candidates_value, list) or len(candidates_value) < 2:
            raise _fail(
                DATASET_TOO_FEW_CANDIDATES,
                f"CLAMBSQL index={index_value} 候选数少于 2",
            )
        if any(not isinstance(sql, str) or not sql.strip() for sql in candidates_value):
            raise _fail(
                DATASET_MISSING_REQUIRED_FIELD,
                f"CLAMBSQL index={index_value} 含空或非字符串候选 SQL",
            )
        gold_ambiguity = item.get("gold_ambiguity")
        if not isinstance(gold_ambiguity, dict):
            raise _fail(
                DATASET_MISSING_REQUIRED_FIELD,
                f"CLAMBSQL index={index_value} 缺少 object 类型 gold_ambiguity",
            )

        sqlite_path = (source_root.joinpath(*db_file.parts)).resolve()
        try:
            sqlite_path.relative_to(source_root)
        except ValueError:
            raise _fail(
                DATASET_UNSUPPORTED_FORMAT,
                f"CLAMBSQL db_file 逃逸 source_database_root: {db_file_value!r}",
            ) from None
        if not sqlite_path.is_file():
            raise _fail(
                DATASET_FILE_NOT_FOUND,
                f"CLAMBSQL 数据库文件不存在: {sqlite_path}",
            )

        schema = catalog_cache.get(db_file_value)
        if schema is None:
            schema = _sqlite_catalog(
                sqlite_path,
                db_file=db_file_value,
                dialect_id="sqlite" if engine_name == "SQLITE" else "duckdb",
            )
            catalog_cache[db_file_value] = schema
        catalog, temporal_columns = schema

        request_id = f"clambsql-{index_value}"
        candidates = tuple(
            {"candidate_id": f"{request_id}-c{candidate_index}", "sql_text": sql}
            for candidate_index, sql in enumerate(candidates_value)
        )
        database_suffix = ".sqlite" if engine_name == "SQLITE" else ".duckdb"
        database_path = (snapshot_root / db_file.with_suffix(database_suffix)).as_posix()
        request = StandardRequest(
            request_id=request_id,
            question_text=question,
            catalog=catalog,
            database_path=database_path,
            candidates=candidates,
            # CLAMBSQL 的发布问题为英文；语言进入运行输入与渲染身份，不能
            # 默默使用 StandardRequest 的中文通用默认值。
            locale=_CLAMBSQL_LOCALE,
            engine_name=engine_name,
            temporal_columns=temporal_columns,
        )
        records.append(
            ClambsqlRecord(
                request=request,
                index=index_value,
                db_id=db_id,
                ambig_type=ambig_type,
                db_file=db_file_value,
                schema_without_content=schema_without_content,
                schema_with_content=schema_with_content,
                gold_ambiguity=gold_ambiguity,
                clarification_context=clarification_context,
                clear_ambiguity=clear_ambiguity,
                gold_query=gold_query,
            )
        )

    manifest = DatasetManifest(
        source="CLAMBSQL",
        license=license,
        version=revision,
        file_sha256=sha256_of_file(data_path),
        database_snapshot=snapshot_root.as_posix(),
        split="unspecified",
        adapter_version=ADAPTER_VERSION,
        snapshot_profile="native_sqlite" if engine_name == "SQLITE" else "legacy",
        engine_name=engine_name,
    )
    return ClambsqlDataset(records=tuple(records), manifest=manifest)
