"""跨进程串行化的 JSONL Event Log。

Event Log 是 Memory 的唯一持久化事实来源。写入使用 append-only 文件句柄、
OS 级锁和 fsync；重复 answer event 幂等跳过，ID 内容冲突则拒绝，避免重放时
重复增加偏好权重或把不同回答折叠成同一条事件。
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cadence.memory.model import PreferenceEventV1, PreferenceMemoryError

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _acquire_os_lock(stream: Any) -> None:
    """获取平台文件锁；错误在外层统一转为稳定 Memory reason code。"""

    if os.name == "nt":
        import msvcrt

        # 先锁偏移 0，再初始化锁字节；先写后锁会让两个首次并发进程竞争
        # 写入空锁文件，在 Windows 上可触发共享冲突并误报存储失败。
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
    else:
        fcntl: Any = __import__("fcntl")
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)


def _release_os_lock(stream: Any) -> None:
    """释放平台文件锁；失败时调用方不得误报为成功完成的存储操作。"""

    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl: Any = __import__("fcntl")
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """拒绝重复 JSON 成员名，避免同一账本行因解析器策略产生歧义。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _loads_unique_json(raw: str | bytes) -> Any:
    """所有账本读取入口共用重复成员检查。"""

    return json.loads(raw, object_pairs_hook=_unique_json_object)


@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """OS 级排它锁防止不同进程并发追加；锁文件不是偏好事实来源。"""

    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        thread_key = str(lock_path.resolve())
        stream = lock_path.open("a+b")
    except OSError as exc:
        raise PreferenceMemoryError("MEMORY_STORE_LOCK_FAILED") from exc
    with _LOCKS_GUARD:
        thread_lock = _LOCKS.setdefault(thread_key, threading.RLock())
    with thread_lock, stream:
        try:
            _acquire_os_lock(stream)
        except OSError as exc:
            raise PreferenceMemoryError("MEMORY_STORE_LOCK_FAILED") from exc
        try:
            yield
        finally:
            try:
                _release_os_lock(stream)
            except OSError as exc:
                raise PreferenceMemoryError("MEMORY_STORE_LOCK_FAILED") from exc


class MemoryEventLog:
    """提供验证、读取和幂等追加；不持久化可重建的偏好汇总。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def read_all(self) -> tuple[PreferenceEventV1, ...]:
        """读取完整 Event Log；不完整末行/无效 JSON fail-closed。"""

        with _file_lock(self.lock_path):
            return self._read_all_locked()

    def append(self, event: PreferenceEventV1) -> str:
        """追加一个 Event；相同幂等键及内容返回 DUPLICATE。"""

        # dataclass 可由调用方直接构造；写入边界再次按封闭契约校验，避免无效对象
        # 成为不可重放的账本记录，直到下次读取才暴露损坏。
        event = PreferenceEventV1.from_mapping(event.to_mapping())
        with _file_lock(self.lock_path):
            self._recover_incomplete_tail_locked()
            events = self._read_all_locked()
            key = (event.namespace, event.scope.user_id, event.event_id)
            for previous in events:
                previous_key = (
                    previous.namespace,
                    previous.scope.user_id,
                    previous.event_id,
                )
                if previous_key == key:
                    if previous.to_mapping() != event.to_mapping():
                        raise PreferenceMemoryError("MEMORY_EVENT_ID_CONFLICT")
                    return "DUPLICATE"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            encoded = (
                json.dumps(
                    event.to_mapping(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_APPEND | os.O_WRONLY,
                    0o600,
                )
                offset = 0
                while offset < len(encoded):
                    written = os.write(descriptor, encoded[offset:])
                    if written <= 0:
                        raise OSError("append returned a non-positive byte count")
                    offset += written
                os.fsync(descriptor)
            except OSError as exc:
                raise PreferenceMemoryError("MEMORY_STORE_WRITE_FAILED") from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            return "APPENDED"

    def _recover_incomplete_tail_locked(self) -> None:
        """append 前修复未完成的末行；完整行保持逐字不变。"""

        if not self.path.exists():
            return
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise PreferenceMemoryError("MEMORY_STORE_READ_FAILED") from exc
        if not raw or raw.endswith(b"\n"):
            return

        last_newline = raw.rfind(b"\n")
        complete_length = last_newline + 1
        tail = raw[complete_length:]
        try:
            payload = _loads_unique_json(tail.decode("utf-8"))
            PreferenceEventV1.from_mapping(payload)
        except (UnicodeDecodeError, ValueError, PreferenceMemoryError):
            # 只有末尾未以 LF 结束的尾段可丢弃；此前已完整记录不回滚。
            self._truncate_locked(complete_length)
        else:
            # 进程可能在完整 JSON 写完、LF 写入前退出；验证结构后补齐记录边界。
            try:
                with self.path.open("ab") as stream:
                    stream.write(b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as exc:
                raise PreferenceMemoryError("MEMORY_STORE_WRITE_FAILED") from exc

    def _truncate_locked(self, size: int) -> None:
        """仅在持有 OS 锁时截断未完成尾行，并将恢复动作同步到磁盘。"""

        try:
            with self.path.open("r+b") as stream:
                stream.truncate(size)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise PreferenceMemoryError("MEMORY_STORE_WRITE_FAILED") from exc

    def _read_all_locked(self) -> tuple[PreferenceEventV1, ...]:
        if not self.path.exists():
            return ()
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise PreferenceMemoryError("MEMORY_STORE_READ_FAILED") from exc
        if raw and not raw.endswith(b"\n"):
            raise PreferenceMemoryError("MEMORY_STORE_TRUNCATED")
        events: list[PreferenceEventV1] = []
        seen: dict[tuple[str, str, str], PreferenceEventV1] = {}
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line:
                raise PreferenceMemoryError("MEMORY_STORE_CORRUPT", str(line_number))
            try:
                payload = _loads_unique_json(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise PreferenceMemoryError("MEMORY_STORE_CORRUPT", str(line_number)) from exc
            event = PreferenceEventV1.from_mapping(payload)
            key = (event.namespace, event.scope.user_id, event.event_id)
            previous = seen.get(key)
            if previous is not None:
                if previous.to_mapping() != event.to_mapping():
                    raise PreferenceMemoryError("MEMORY_EVENT_ID_CONFLICT")
                raise PreferenceMemoryError("MEMORY_STORE_DUPLICATE_EVENT")
            seen[key] = event
            events.append(event)
        return tuple(events)
