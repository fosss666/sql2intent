"""以只读 SQLite 执行结果为 CLAMBSQL 的预测与 Gold 评分。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

SQL_TIMEOUT_SECONDS = 30


def _typed_value(value: object) -> str:
    """为 SQL 结果加类型前缀，避免数值与文本等值被误判为一致。"""

    if value is None:
        return "NULL:"
    if isinstance(value, bool):
        return "BOOL:T" if value else "BOOL:F"
    if isinstance(value, int):
        return f"INT:{value}"
    if isinstance(value, float):
        return f"FLOAT:{value!r}"
    if isinstance(value, bytes):
        return f"BYTES:{value.hex()}"
    if isinstance(value, str):
        return f"STR:{value}"
    return f"OBJ:{type(value).__name__}:{value!r}"


def _result_signature(database: Path, sql: str) -> str:
    """比较时忽略行顺序，保留列顺序、重复行和 SQLite 标量类型。"""

    uri = f"{database.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=SQL_TIMEOUT_SECONDS)
    deadline = time.monotonic() + SQL_TIMEOUT_SECONDS
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        rows = connection.execute(sql).fetchall()
    finally:
        connection.close()
    typed_rows = sorted(tuple(_typed_value(value) for value in row) for row in rows)
    payload = json.dumps(typed_rows, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()
