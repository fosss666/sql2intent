"""SQLite 持久化与 sqlite-vec 派生索引。

事件 JSON 是唯一事实来源；规范化键列和 encoder 向量只用于索引，可以从事件重建。
SQLite 事务负责跨进程追加/查重，SQL scope 索引和 vec0 分区避免全库事件扫描。
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import sqlite3
import unicodedata
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from cadence.memory.model import (
    _FIXED_SEMANTIC_TOP_K,
    MemoryScope,
    PreferenceEventV1,
    PreferenceMemoryError,
)

_SEPARATOR = re.compile(r"[^\w]+", re.UNICODE)
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_SQLITE_TIMEOUT_SECONDS = 30.0


def _normalize_key(value: str) -> str:
    """与 MemoryEngine 的键规范化一致，避免存储模块反向导入评分器。"""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return _SEPARATOR.sub(" ", normalized).strip()


def _scope_parts(scope: MemoryScope) -> tuple[str, str, str]:
    """召回按 namespace/database/Schema 共享；user_id 只保留在事件来源记录。"""

    return (scope.namespace, scope.database_id, scope.schema_id)


def _scope_digest(scope: MemoryScope) -> str:
    """为 vec0 partition key 生成跨进程稳定且无分隔符歧义的摘要。"""

    encoded = json.dumps(
        _scope_parts(scope), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _time_key(value: datetime, field: str) -> str:
    """转成固定宽度 UTC 文本，使 SQLite 字符串比较等价于时间顺序。"""

    if value.tzinfo is None or value.utcoffset() is None:
        raise PreferenceMemoryError("MEMORY_TIME_INVALID", field)
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _vector_values(vector: Sequence[float]) -> tuple[float, ...]:
    """验证缓存向量；坏向量不能被当成低相关记录写入检索索引。"""

    if isinstance(vector, (str, bytes, bytearray)):
        raise PreferenceMemoryError("MEMORY_ENCODER_OUTPUT_INVALID")
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PreferenceMemoryError("MEMORY_ENCODER_OUTPUT_INVALID") from exc
    if (
        not values
        or any(not math.isfinite(value) for value in values)
        or math.hypot(*values) == 0.0
    ):
        raise PreferenceMemoryError("MEMORY_ENCODER_OUTPUT_INVALID")
    return values


def _vector_json(values: Sequence[float]) -> str:
    return json.dumps(list(values), separators=(",", ":"), allow_nan=False)


def _encoder_table_name(encoder_id: str, dimension: int) -> str:
    digest = hashlib.sha256(encoder_id.encode("utf-8")).hexdigest()
    return f"vec_{digest}_d{dimension}"


class SQLiteMemoryStore:
    """保存 PreferenceEventV1 并维护可重建的 sqlite-vec cosine 索引。

    数据库文件可由多个进程共享。规范事件 payload 是唯一事实来源；向量缓存按
    encoder_id 和维度隔离，丢失时通过 ``events_missing_embeddings`` 重建。
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.path,
                timeout=_SQLITE_TIMEOUT_SECONDS,
                isolation_level=None,
            )
        except (OSError, sqlite3.Error) as exc:
            raise PreferenceMemoryError("MEMORY_STORE_OPEN_FAILED") from exc

        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(f"PRAGMA busy_timeout={int(_SQLITE_TIMEOUT_SECONDS * 1000)}")
            try:
                sqlite_vec = importlib.import_module("sqlite_vec")
                connection.enable_load_extension(True)
                sqlite_vec.load(connection)
            except (ImportError, OSError, sqlite3.Error) as exc:
                raise PreferenceMemoryError("MEMORY_VECTOR_EXTENSION_UNAVAILABLE") from exc
            finally:
                try:
                    connection.enable_load_extension(False)
                except sqlite3.Error as exc:
                    raise PreferenceMemoryError("MEMORY_VECTOR_EXTENSION_UNAVAILABLE") from exc
            yield connection
        finally:
            connection.close()

    @staticmethod
    @contextmanager
    def _write_transaction(connection: sqlite3.Connection) -> Iterator[None]:
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()

    def _initialize(self) -> None:
        try:
            with self._connection() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=FULL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS memory_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        namespace TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        database_id TEXT NOT NULL,
                        schema_id TEXT NOT NULL,
                        ambiguity_key_norm TEXT NOT NULL,
                        context_key_norm TEXT,
                        occurred_at TEXT NOT NULL,
                        preference_value_id TEXT NOT NULL,
                        event_id TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        UNIQUE(namespace, user_id, event_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_memory_shared_scope_time
                        ON memory_events(
                            namespace, database_id, schema_id,
                            occurred_at, event_id
                        );
                    CREATE INDEX IF NOT EXISTS idx_memory_shared_exact
                        ON memory_events(
                            namespace, database_id, schema_id,
                            ambiguity_key_norm, context_key_norm, occurred_at, event_id
                        );
                    CREATE TABLE IF NOT EXISTS memory_encoder_registry (
                        encoder_id TEXT PRIMARY KEY,
                        dimension INTEGER,
                        table_name TEXT UNIQUE
                    );
                    CREATE TABLE IF NOT EXISTS memory_embedding_scopes (
                        encoder_id TEXT NOT NULL,
                        scope_key TEXT NOT NULL,
                        PRIMARY KEY(encoder_id, scope_key),
                        FOREIGN KEY(encoder_id)
                            REFERENCES memory_encoder_registry(encoder_id)
                    );
                    CREATE TABLE IF NOT EXISTS memory_embedding_queue (
                        encoder_id TEXT NOT NULL,
                        scope_key TEXT NOT NULL,
                        event_rowid INTEGER NOT NULL,
                        PRIMARY KEY(encoder_id, scope_key, event_rowid),
                        FOREIGN KEY(encoder_id)
                            REFERENCES memory_encoder_registry(encoder_id),
                        FOREIGN KEY(event_rowid)
                            REFERENCES memory_events(id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_memory_embedding_queue_scope
                        ON memory_embedding_queue(encoder_id, scope_key, event_rowid);
                    """
                )
        except PreferenceMemoryError:
            raise
        except sqlite3.Error as exc:
            raise PreferenceMemoryError("MEMORY_STORE_WRITE_FAILED") from exc

    @staticmethod
    def _validated_event(event: PreferenceEventV1) -> PreferenceEventV1:
        if not isinstance(event, PreferenceEventV1):
            raise PreferenceMemoryError("MEMORY_EVENT_SCHEMA_INVALID", "event type")
        return PreferenceEventV1.from_mapping(event.to_mapping())

    @staticmethod
    def _payload(event: PreferenceEventV1) -> str:
        return json.dumps(
            event.to_mapping(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @staticmethod
    def _decode_payload(payload: str) -> PreferenceEventV1:
        try:
            value = json.loads(payload)
            return PreferenceEventV1.from_mapping(value)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, PreferenceMemoryError):
                raise
            raise PreferenceMemoryError("MEMORY_STORE_CORRUPT") from exc

    @staticmethod
    def _where_scope() -> str:
        return "namespace=? AND database_id=? AND schema_id=?"

    @staticmethod
    def _where_event_scope() -> str:
        """指定某条事件时仍使用其来源 user_id，避免同 ID 跨用户误写向量。"""

        return "namespace=? AND user_id=? AND database_id=? AND schema_id=?"

    @staticmethod
    def _row_payloads(rows: Sequence[sqlite3.Row]) -> tuple[PreferenceEventV1, ...]:
        return tuple(SQLiteMemoryStore._decode_payload(row[0]) for row in rows)

    @staticmethod
    def _event_columns(event: PreferenceEventV1) -> tuple[object, ...]:
        return (
            event.namespace,
            event.scope.user_id,
            event.scope.database_id,
            event.scope.schema_id,
            _normalize_key(event.ambiguity_key),
            _normalize_key(event.context_key) if event.context_key is not None else None,
            _time_key(event.occurred_at, "occurred_at"),
            event.preference_value_id,
            event.event_id,
            SQLiteMemoryStore._payload(event),
        )

    def append(self, event: PreferenceEventV1) -> str:
        """幂等追加事件；同一唯一键不同 payload 稳定失败。"""

        event = self._validated_event(event)
        try:
            with self._connection() as connection, self._write_transaction(connection):
                previous = connection.execute(
                    """
                        SELECT payload FROM memory_events
                        WHERE namespace=? AND user_id=? AND event_id=?
                        """,
                    (event.namespace, event.scope.user_id, event.event_id),
                ).fetchone()
                payload = self._payload(event)
                if previous is not None:
                    if previous["payload"] != payload:
                        raise PreferenceMemoryError("MEMORY_EVENT_ID_CONFLICT")
                    return "DUPLICATE"

                cursor = connection.execute(
                    """
                        INSERT INTO memory_events(
                            namespace, user_id, database_id, schema_id,
                            ambiguity_key_norm, context_key_norm, occurred_at,
                            preference_value_id, event_id, payload
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                    self._event_columns(event),
                )
                if cursor.lastrowid is None:
                    raise PreferenceMemoryError("MEMORY_STORE_WRITE_FAILED")
                event_rowid = cursor.lastrowid
                scope_key = _scope_digest(event.scope)
                encoders = connection.execute(
                    """
                        SELECT encoder_id FROM memory_embedding_scopes
                        WHERE scope_key=? ORDER BY encoder_id
                        """,
                    (scope_key,),
                ).fetchall()
                connection.executemany(
                    """
                        INSERT OR IGNORE INTO memory_embedding_queue(
                            encoder_id, scope_key, event_rowid
                        ) VALUES (?, ?, ?)
                        """,
                    ((row["encoder_id"], scope_key, event_rowid) for row in encoders),
                )
            return "APPENDED"
        except PreferenceMemoryError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise PreferenceMemoryError("MEMORY_STORE_WRITE_FAILED") from exc

    def read_all(self) -> tuple[PreferenceEventV1, ...]:
        """兼容及诊断接口；在线检索应使用 scope/exact/vector 查询。"""

        try:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT payload FROM memory_events ORDER BY id"
                ).fetchall()
            return self._row_payloads(rows)
        except PreferenceMemoryError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise PreferenceMemoryError("MEMORY_STORE_READ_FAILED") from exc

    def read_scope(
        self,
        scope: MemoryScope,
        *,
        as_of: datetime | None = None,
    ) -> tuple[PreferenceEventV1, ...]:
        """仅从 scope B-tree 范围读取事件，可选地排除未来记录。"""

        try:
            with self._connection() as connection:
                sql = f"SELECT payload FROM memory_events WHERE {self._where_scope()}"
                parameters: list[object] = list(_scope_parts(scope))
                if as_of is not None:
                    sql += " AND occurred_at<=?"
                    parameters.append(_time_key(as_of, "as_of"))
                sql += " ORDER BY id"
                rows = connection.execute(sql, parameters).fetchall()
            return self._row_payloads(rows)
        except PreferenceMemoryError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise PreferenceMemoryError("MEMORY_STORE_READ_FAILED") from exc

    def find_exact(
        self,
        scope: MemoryScope,
        ambiguity_keys: Sequence[str] | set[str],
        context_keys: Sequence[str | None] | set[str | None],
    ) -> tuple[PreferenceEventV1, ...]:
        """用复合 B-tree 索引查规范化歧义键/语境键的任一精确组合。"""

        normalized_ambiguity = sorted(
            {_normalize_key(value) for value in ambiguity_keys if value.strip()}
        )
        normalized_context = sorted(
            {
                _normalize_key(value)
                for value in context_keys
                if value is not None and value.strip()
            }
        )
        include_null = None in context_keys
        if not normalized_ambiguity or (not normalized_context and not include_null):
            return ()

        try:
            with self._connection() as connection:
                clauses = [self._where_scope()]
                parameters: list[object] = list(_scope_parts(scope))
                clauses.append(
                    "ambiguity_key_norm IN ("
                    + ",".join("?" for _ in normalized_ambiguity)
                    + ")"
                )
                parameters.extend(normalized_ambiguity)
                context_clauses: list[str] = []
                if normalized_context:
                    context_clauses.append(
                        "context_key_norm IN ("
                        + ",".join("?" for _ in normalized_context)
                        + ")"
                    )
                    parameters.extend(normalized_context)
                if include_null:
                    context_clauses.append("context_key_norm IS NULL")
                clauses.append("(" + " OR ".join(context_clauses) + ")")
                sql = (
                    "SELECT payload FROM memory_events WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY occurred_at, event_id, id"
                )
                rows = connection.execute(sql, parameters).fetchall()
            return self._row_payloads(rows)
        except PreferenceMemoryError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise PreferenceMemoryError("MEMORY_STORE_READ_FAILED") from exc

    @staticmethod
    def _ensure_encoder(
        connection: sqlite3.Connection,
        encoder_id: str,
    ) -> sqlite3.Row:
        if not isinstance(encoder_id, str) or not encoder_id.strip():
            raise PreferenceMemoryError("MEMORY_CONFIG_INVALID", "encoder_id")
        connection.execute(
            "INSERT OR IGNORE INTO memory_encoder_registry(encoder_id) VALUES (?)",
            (encoder_id,),
        )
        row = connection.execute(
            "SELECT encoder_id, dimension, table_name FROM memory_encoder_registry "
            "WHERE encoder_id=?",
            (encoder_id,),
        ).fetchone()
        if row is None:
            raise PreferenceMemoryError("MEMORY_STORE_CORRUPT", "encoder registry")
        return cast(sqlite3.Row, row)

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _ensure_vector_table(
        connection: sqlite3.Connection,
        encoder_id: str,
        dimension: int,
    ) -> str:
        registry = SQLiteMemoryStore._ensure_encoder(connection, encoder_id)
        expected_name = _encoder_table_name(encoder_id, dimension)
        if registry["dimension"] is not None and registry["dimension"] != dimension:
            raise PreferenceMemoryError("MEMORY_ENCODER_DIMENSION_MISMATCH")
        if registry["table_name"] not in (None, expected_name):
            raise PreferenceMemoryError("MEMORY_STORE_CORRUPT", "encoder table mapping")
        connection.execute(
            "UPDATE memory_encoder_registry SET dimension=?, table_name=? WHERE encoder_id=?",
            (dimension, expected_name, encoder_id),
        )
        # Identifier由 SHA-256 与整数维度组成，不能包含外部输入的 SQL 字符。
        connection.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS \"{expected_name}\" USING vec0("
            f"embedding float[{dimension}] distance_metric=cosine, "
            "scope_key TEXT PARTITION KEY)"
        )
        return expected_name

    def _initialize_embedding_scope(
        self,
        connection: sqlite3.Connection,
        scope: MemoryScope,
        encoder_id: str,
        *,
        repair_missing: bool = True,
    ) -> None:
        registry = self._ensure_encoder(connection, encoder_id)
        scope_key = _scope_digest(scope)
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO memory_embedding_scopes(encoder_id, scope_key)
            VALUES (?, ?)
            """,
            (encoder_id, scope_key),
        )
        table_name: str | None = registry["table_name"]
        vector_table_was_missing = bool(
            table_name and not self._table_exists(connection, table_name)
        )
        if registry["dimension"] is not None:
            table_name = self._ensure_vector_table(
                connection, encoder_id, int(registry["dimension"])
            )
        if (
            cursor.rowcount == 0
            and not vector_table_was_missing
            and not repair_missing
        ):
            # 追加时已把新事件放入 queue；常规查询只取增量队列，避免每次扫完整 scope。
            # 只有显式完整性修复或索引表重建时，才左连接核对全部事件与向量行。
            return

        scope_sql = "e.namespace=? AND e.database_id=? AND e.schema_id=?"
        scope_values = _scope_parts(scope)
        if table_name and self._table_exists(connection, table_name):
            rows = connection.execute(
                f"""
                SELECT e.id FROM memory_events AS e
                LEFT JOIN \"{table_name}\" AS v
                    ON v.rowid=e.id AND v.scope_key=?
                WHERE {scope_sql} AND v.rowid IS NULL
                ORDER BY e.id
                """,
                (scope_key, *scope_values),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id FROM memory_events
                WHERE namespace=? AND database_id=? AND schema_id=?
                ORDER BY id
                """,
                scope_values,
            ).fetchall()
        connection.executemany(
            """
            INSERT OR IGNORE INTO memory_embedding_queue(
                encoder_id, scope_key, event_rowid
            ) VALUES (?, ?, ?)
            """,
            ((encoder_id, scope_key, row["id"]) for row in rows),
        )

    def events_missing_embeddings(
        self,
        scope: MemoryScope,
        encoder_id: str,
        *,
        repair_missing: bool = False,
    ) -> tuple[PreferenceEventV1, ...]:
        """读取增量编码队列；可选扫描当前 scope 修复丢失的派生向量行。"""

        try:
            with self._connection() as connection, self._write_transaction(connection):
                self._initialize_embedding_scope(
                    connection,
                    scope,
                    encoder_id,
                    repair_missing=repair_missing,
                )
                rows = connection.execute(
                    """
                        SELECT e.payload FROM memory_embedding_queue AS q
                        JOIN memory_events AS e ON e.id=q.event_rowid
                        WHERE q.encoder_id=? AND q.scope_key=?
                        ORDER BY e.id
                        """,
                    (encoder_id, _scope_digest(scope)),
                ).fetchall()
            return self._row_payloads(rows)
        except PreferenceMemoryError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise PreferenceMemoryError("MEMORY_STORE_READ_FAILED") from exc

    @staticmethod
    def _query_knn_rows(
        connection: sqlite3.Connection,
        table_name: str,
        vector_json: str,
        requested_k: int,
        scope_key: str,
    ) -> list[sqlite3.Row]:
        """按指定 KNN 上限查询，供边界并列扩取逻辑重复调用。"""

        return connection.execute(
            f"""
            SELECT e.payload, v.distance FROM \"{table_name}\" AS v
            JOIN memory_events AS e ON e.id=v.rowid
            WHERE v.embedding MATCH ? AND k=? AND v.scope_key=?
            ORDER BY v.distance
            """,
            (vector_json, requested_k, scope_key),
        ).fetchall()

    def upsert_embedding(
        self,
        event_id: str,
        scope: MemoryScope,
        encoder_id: str,
        vector: Sequence[float],
    ) -> None:
        """写入/替换派生向量；原始事件 payload 不因 encoder 变化而修改。"""

        values = _vector_values(vector)
        vector_text = _vector_json(values)
        try:
            with self._connection() as connection, self._write_transaction(connection):
                row = connection.execute(
                    f"SELECT id FROM memory_events WHERE {self._where_event_scope()} "
                    "AND event_id=?",
                    (
                        scope.namespace,
                        scope.user_id,
                        scope.database_id,
                        scope.schema_id,
                        event_id,
                    ),
                ).fetchone()
                if row is None:
                    raise PreferenceMemoryError("MEMORY_EVENT_SCOPE_MISMATCH")
                self._initialize_embedding_scope(
                    connection, scope, encoder_id, repair_missing=False
                )
                table_name = self._ensure_vector_table(
                    connection, encoder_id, len(values)
                )
                connection.execute(
                    f"INSERT OR REPLACE INTO \"{table_name}\""
                    "(rowid, embedding, scope_key) VALUES (?, ?, ?)",
                    (int(row["id"]), vector_text, _scope_digest(scope)),
                )
                connection.execute(
                    """
                        DELETE FROM memory_embedding_queue
                        WHERE encoder_id=? AND scope_key=? AND event_rowid=?
                        """,
                    (encoder_id, _scope_digest(scope), int(row["id"])),
                )
        except PreferenceMemoryError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise PreferenceMemoryError("MEMORY_STORE_WRITE_FAILED") from exc

    def search_similar(
        self,
        scope: MemoryScope,
        encoder_id: str,
        query_vector: Sequence[float],
        *,
        top_k: int = _FIXED_SEMANTIC_TOP_K,
    ) -> list[tuple[PreferenceEventV1, float]]:
        """返回 scope 内 cosine KNN 的前 top-k 条，不应用 T_r。"""

        if top_k <= 0:
            raise PreferenceMemoryError("MEMORY_CONFIG_INVALID", "semantic_top_k")
        values = _vector_values(query_vector)
        try:
            with self._connection() as connection:
                registry = connection.execute(
                    "SELECT dimension, table_name FROM memory_encoder_registry "
                    "WHERE encoder_id=?",
                    (encoder_id,),
                ).fetchone()
                if registry is None or registry["dimension"] is None:
                    return []
                if int(registry["dimension"]) != len(values):
                    raise PreferenceMemoryError("MEMORY_ENCODER_DIMENSION_MISMATCH")
                table_name = str(registry["table_name"])
                if not self._table_exists(connection, table_name):
                    raise PreferenceMemoryError("MEMORY_VECTOR_INDEX_MISSING")
                scope_key = _scope_digest(scope)
                count = int(
                    connection.execute(
                        f'SELECT count(*) FROM "{table_name}" WHERE scope_key=?',
                        (scope_key,),
                    ).fetchone()[0]
                )
                if count == 0:
                    return []
                requested_k = min(count, top_k + 1)
                vector_json = _vector_json(values)
                while True:
                    rows = self._query_knn_rows(
                        connection,
                        table_name,
                        vector_json,
                        requested_k,
                        scope_key,
                    )
                    if requested_k >= count or len(rows) <= top_k:
                        break
                    boundary_distance = float(rows[top_k - 1]["distance"])
                    if float(rows[-1]["distance"]) > boundary_distance:
                        break
                    # vec0 不承诺相似度并列项的截断顺序；先取全边界并列组，
                    # 再按时间与 event_id 稳定排序，否则相同向量会令 top-k 跨运行漂移。
                    requested_k = min(count, max(requested_k + 1, requested_k * 2))
            matches: list[tuple[PreferenceEventV1, float]] = []
            for row in rows:
                event = self._decode_payload(row["payload"])
                similarity = max(-1.0, min(1.0, 1.0 - float(row["distance"])))
                matches.append((event, similarity))
            matches.sort(
                key=lambda item: (
                    -item[1],
                    -item[0].occurred_at.timestamp(),
                    item[0].event_id,
                )
            )
            return matches[:top_k]
        except PreferenceMemoryError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise PreferenceMemoryError("MEMORY_STORE_READ_FAILED") from exc
