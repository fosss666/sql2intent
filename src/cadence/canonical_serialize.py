"""生成用于候选身份和事件 ID 的确定性 JSON 字节。

输入仅接受 null、bool、string、array 和 object；字符串按 NFC 规范化，object key
按规范顺序编码。拒绝 number、surrogate 和规范化后冲突的 key。
"""

from __future__ import annotations

import hashlib
import unicodedata

# 序列化失败码。
CANONICAL_INPUT_INVALID_JSON = "CANONICAL_INPUT_INVALID_JSON"
CANONICAL_KEY_COLLISION_AFTER_NFC = "CANONICAL_KEY_COLLISION_AFTER_NFC"
CANONICAL_NON_SCALAR_UNICODE = "CANONICAL_NON_SCALAR_UNICODE"
CANONICAL_NUMBER_TOKEN_FORBIDDEN = "CANONICAL_NUMBER_TOKEN_FORBIDDEN"

# 拒绝 JSON 值域之外的 Python 输入，避免把非法对象隐式转换成有效序列化值。
CANONICAL_NON_STRING_KEY = "CANONICAL_NON_STRING_KEY"
CANONICAL_UNSUPPORTED_TYPE = "CANONICAL_UNSUPPORTED_TYPE"


class CanonicalSerializeError(ValueError):
    """canonical serializer 输入错误；reason_code 稳定分类错误，detail 提供诊断上下文。"""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(reason_code if not detail else f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


def _escape_string(s: str) -> str:
    """按 string_rules 转义并加引号；s 已 NFC 化且不含 surrogate。"""
    out: list[str] = []
    for ch in s:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\r":
            out.append("\\r")
        elif o < 0x20:
            out.append(f"\\u{o:04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _reject_non_scalar(s: str) -> None:
    """Unicode scalar 域校验：surrogate 拒绝（CANONICAL_NON_SCALAR_UNICODE）。"""
    if any(0xD800 <= ord(c) <= 0xDFFF for c in s):
        raise CanonicalSerializeError(CANONICAL_NON_SCALAR_UNICODE)


def canonicalize_to_text(o: object) -> str:
    """递归把 Python 对象编码为 canonical serializer 文本（不含外层字节）。

    fail-closed：任何无法唯一确定为合法 canonical serializer 值域的输入都拒绝，
    不猜测、不补默认。
    """
    if o is None:
        return "null"
    if o is True:
        return "true"
    if o is False:
        return "false"
    if isinstance(o, str):
        n = unicodedata.normalize("NFC", o)
        _reject_non_scalar(n)
        return _escape_string(n)
    # bool 已在上方拦截；其余 int/float 一律禁止（number token forbidden）。
    if isinstance(o, int):
        raise CanonicalSerializeError(CANONICAL_NUMBER_TOKEN_FORBIDDEN)
    if isinstance(o, float):
        raise CanonicalSerializeError(CANONICAL_NUMBER_TOKEN_FORBIDDEN)
    if isinstance(o, list):
        return "[" + ",".join(canonicalize_to_text(x) for x in o) + "]"
    if isinstance(o, dict):
        norm_keys: list[str] = []
        seen: set[str] = set()
        for k in o:
            if not isinstance(k, str):
                raise CanonicalSerializeError(CANONICAL_NON_STRING_KEY)
            nk = unicodedata.normalize("NFC", k)
            _reject_non_scalar(nk)
            if nk in seen:
                # 两个原始 key 经 NFC 后碰撞：拒绝（CANONICAL_KEY_COLLISION_AFTER_NFC）。
                raise CanonicalSerializeError(CANONICAL_KEY_COLLISION_AFTER_NFC)
            seen.add(nk)
            norm_keys.append(nk)
        # 按 NFC 后码点升序（Python str 比较即按 Unicode code point）。
        norm_keys.sort()
        return "{" + ",".join(
            _escape_string(nk) + ":" + canonicalize_to_text(o[_orig_key(o, nk)])
            for nk in norm_keys
        ) + "}"
    raise CanonicalSerializeError(
        CANONICAL_UNSUPPORTED_TYPE, type(o).__name__)


def _orig_key(d: dict[object, object], nk: str) -> object:
    """NFC 后的 key -> 原始 dict key（唯一；碰撞已在上方拒绝）。"""
    for k in d:
        if isinstance(k, str) and unicodedata.normalize("NFC", k) == nk:
            return k
    raise CanonicalSerializeError(CANONICAL_NON_STRING_KEY)


def canonical_bytes(o: object) -> bytes:
    """canonical serializer 输出字节（UTF-8、无 BOM、字符串外无空白、无尾换行）。"""
    return canonicalize_to_text(o).encode("utf-8")


def sha256_canonical(o: object) -> str:
    """对 canonical serializer 字节取 SHA-256 小写 64-hex（科研 hash 统一入口）。"""
    return hashlib.sha256(canonical_bytes(o)).hexdigest()
