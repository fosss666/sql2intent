"""为 CADENCE callback 提供 OpenAI-compatible Chat Completions 传输。

本模块只负责配置解析后的 HTTP 传输，不生成提示词，也不解释模型输出。模型选择、
温度和响应格式来自版本化 runtime config；API key 只在请求头使用，不写入产物。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from cadence.runtime_config import (
    RuntimeConfig,
    RuntimeConfigError,
    require_api_key_env,
    resolve_runtime_role,
)


class CadenceProviderError(RuntimeError):
    """传输失败使用稳定 reason code，异常不包含 API key 或响应正文。"""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def call_role(
    runtime_config: RuntimeConfig,
    role: str,
    messages: Sequence[Mapping[str, str]],
) -> tuple[str, dict[str, Any]]:
    """通过 runtime 中冻结的 role 发起单次同步请求。"""

    try:
        resolved = resolve_runtime_role(runtime_config, role)
        if resolved.network_policy != "online":
            raise CadenceProviderError("RUNTIME_CONFIG_NETWORK_POLICY_OFFLINE")
        if not resolved.enabled:
            raise CadenceProviderError("RUNTIME_CONFIG_ROLE_DISABLED")
        if resolved.provider_type != "openai_compatible" or not resolved.base_url:
            raise CadenceProviderError("RUNTIME_CONFIG_PROVIDER_NOT_HTTP")
        if not resolved.model or not resolved.api_key_env:
            raise CadenceProviderError("RUNTIME_CONFIG_PROVIDER_FIELDS_MISSING")
        require_api_key_env(resolved, os.environ)
        api_key = os.environ[resolved.api_key_env]
    except CadenceProviderError:
        raise
    except RuntimeConfigError as exc:
        raise CadenceProviderError(exc.reason_code) from exc

    payload: dict[str, Any] = {
        "model": resolved.model,
        "messages": [dict(message) for message in messages],
        "temperature": resolved.temperature if resolved.temperature is not None else 0.0,
        "max_tokens": resolved.max_tokens if resolved.max_tokens is not None else 1024,
        "response_format": dict(resolved.response_format or {"type": "json_object"}),
    }
    if resolved.provider_id == "deepseek" and resolved.thinking_type is not None:
        if resolved.thinking_type not in {"enabled", "disabled"}:
            raise CadenceProviderError("RUNTIME_CONFIG_THINKING_TYPE_UNSUPPORTED")
        payload["thinking"] = {"type": resolved.thinking_type}
    request = urllib.request.Request(
        resolved.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=resolved.timeout_seconds or 60
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise CadenceProviderError("LLM_CONTENT_INVALID")
    except CadenceProviderError:
        raise
    except urllib.error.HTTPError as exc:
        raise CadenceProviderError("LLM_HTTP_ERROR") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = "LLM_TIMEOUT" if isinstance(exc, TimeoutError) else "LLM_TRANSPORT_ERROR"
        raise CadenceProviderError(reason) from exc
    except (KeyError, IndexError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise CadenceProviderError("LLM_RESPONSE_INVALID") from exc

    content = content.replace(api_key, "[REDACTED_API_KEY]") if api_key else content
    response_id = body.get("id")
    usage = body.get("usage")
    return content, {
        "model_id": resolved.model_id,
        "model": resolved.model,
        "provider_id": resolved.provider_id,
        "provider_response_id": response_id if isinstance(response_id, str) else None,
        "token_usage": dict(usage) if isinstance(usage, Mapping) else None,
    }
