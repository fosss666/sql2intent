"""读取当前 CADENCE provider、模型、角色和运行策略配置。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]


class RuntimeConfigError(ValueError):
    """配置错误及其可供调用方处理的 reason code。"""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(reason_code, detail)
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class RuntimeConfig:
    """解析完成的运行配置快照。"""

    source_path: Path
    data: Mapping[str, Any]
    raw_sha256: str


@dataclass(frozen=True)
class ResolvedModelConfig:
    """一个 role 在 profile 下的有效 provider 和模型设置。"""

    role: str
    profile: str
    model_id: str
    provider_id: str
    provider_type: str
    base_url: str | None
    api_key_env: str | None
    model: str | None
    temperature: float | int | None
    max_tokens: int | None
    thinking_type: str | None
    timeout_seconds: float | int | None
    response_format: Mapping[str, Any] | None
    enabled: bool | None
    network_policy: str


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeConfigError("RUNTIME_CONFIG_SCHEMA_INVALID", f"{field} 必须为 object")
    return value


def _validate_references(data: Mapping[str, Any]) -> None:
    providers = _mapping(data.get("providers"), "providers")
    models = _mapping(data.get("models"), "models")
    roles = _mapping(data.get("roles"), "roles")
    profiles = _mapping(data.get("profiles"), "profiles")
    default_profile = data.get("default_profile")
    if not isinstance(default_profile, str) or default_profile not in profiles:
        raise RuntimeConfigError("RUNTIME_CONFIG_UNKNOWN_PROFILE", str(default_profile))

    for model_id, raw_model in models.items():
        model = _mapping(raw_model, f"models.{model_id}")
        provider_id = model.get("provider")
        if not isinstance(provider_id, str) or provider_id not in providers:
            raise RuntimeConfigError(
                "RUNTIME_CONFIG_PROVIDER_NOT_FOUND", f"models.{model_id}"
            )

    def validate_role_map(role_map: Mapping[str, Any], owner: str) -> None:
        for role, model_id in role_map.items():
            if not isinstance(model_id, str) or model_id not in models:
                raise RuntimeConfigError(
                    "RUNTIME_CONFIG_MODEL_NOT_FOUND", f"{owner}.{role}"
                )

    validate_role_map(roles, "roles")
    for profile_name, raw_profile in profiles.items():
        profile = _mapping(raw_profile, f"profiles.{profile_name}")
        profile_roles = _mapping(profile.get("roles"), f"profiles.{profile_name}.roles")
        validate_role_map(profile_roles, f"profiles.{profile_name}.roles")
        policy = profile.get("network_policy")
        if not isinstance(policy, str) or not policy:
            raise RuntimeConfigError(
                "RUNTIME_CONFIG_SCHEMA_INVALID", f"profiles.{profile_name}.network_policy"
            )

    memory = data.get("memory")
    if memory is not None:
        memory_config = _mapping(memory, "memory")
        enabled = memory_config.get("enabled")
        if not isinstance(enabled, bool):
            raise RuntimeConfigError("RUNTIME_CONFIG_SCHEMA_INVALID", "memory.enabled")
        storage = _mapping(memory_config.get("storage"), "memory.storage")
        if not enabled and storage.get("path") is not None:
            raise RuntimeConfigError(
                "RUNTIME_CONFIG_MEMORY_INVALID", "disabled memory requires storage.path=null"
            )


def load_runtime_config(path: Path) -> RuntimeConfig:
    """读取并校验 YAML 配置。"""

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise RuntimeConfigError("RUNTIME_CONFIG_FILE_NOT_FOUND", str(source_path))
    try:
        raw_bytes = source_path.read_bytes()
    except OSError as exc:
        raise RuntimeConfigError("RUNTIME_CONFIG_FILE_NOT_FOUND", str(source_path)) from exc
    return _load_runtime_config_bytes(raw_bytes, source_path)


def load_runtime_config_from_text(
    raw_text: str, *, source_path: Path | None = None
) -> RuntimeConfig:
    """解析随运行冻结的配置文本。"""

    text_path = source_path or Path("<frozen-runtime-config>")
    try:
        raw_bytes = raw_text.encode("utf-8")
    except UnicodeError as exc:
        raise RuntimeConfigError("RUNTIME_CONFIG_ENCODING_INVALID") from exc
    return _load_runtime_config_bytes(raw_bytes, text_path)


def _load_runtime_config_bytes(raw_bytes: bytes, source_path: Path) -> RuntimeConfig:
    try:
        raw: object = yaml.safe_load(raw_bytes.decode("utf-8"))
    except yaml.YAMLError as exc:
        raise RuntimeConfigError("RUNTIME_CONFIG_INVALID_YAML", str(source_path)) from exc
    except UnicodeError as exc:
        raise RuntimeConfigError("RUNTIME_CONFIG_ENCODING_INVALID", str(source_path)) from exc
    if not isinstance(raw, Mapping):
        raise RuntimeConfigError("RUNTIME_CONFIG_DOCUMENT_INVALID", str(source_path))
    data = cast(Mapping[str, Any], raw)
    _validate_references(data)
    return RuntimeConfig(
        source_path=source_path,
        data=data,
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def resolve_memory_runtime_config(config: RuntimeConfig) -> dict[str, Any] | None:
    """返回 Memory 配置；未配置时表示关闭。"""

    memory = config.data.get("memory")
    return dict(_mapping(memory, "memory")) if memory is not None else None


def resolve_runtime_role(
    config: RuntimeConfig,
    role: str,
    profile: str | None = None,
) -> ResolvedModelConfig:
    """按 profile、role、model 和 provider 解析一个运行角色。"""

    selected_profile = profile or str(config.data["default_profile"])
    profiles = _mapping(config.data["profiles"], "profiles")
    profile_data = _mapping(profiles.get(selected_profile), f"profiles.{selected_profile}")
    profile_roles = _mapping(profile_data["roles"], f"profiles.{selected_profile}.roles")
    roles = _mapping(config.data["roles"], "roles")
    model_id = profile_roles.get(role, roles.get(role))
    if not isinstance(model_id, str):
        raise RuntimeConfigError("RUNTIME_CONFIG_UNKNOWN_ROLE", role)

    models = _mapping(config.data["models"], "models")
    model_data = _mapping(models[model_id], f"models.{model_id}")
    provider_id = str(model_data["provider"])
    providers = _mapping(config.data["providers"], "providers")
    provider_data = _mapping(providers[provider_id], f"providers.{provider_id}")
    return ResolvedModelConfig(
        role=role,
        profile=selected_profile,
        model_id=model_id,
        provider_id=provider_id,
        provider_type=str(provider_data["type"]),
        base_url=provider_data.get("base_url"),
        api_key_env=provider_data.get("api_key_env"),
        model=model_data.get("model"),
        temperature=model_data.get("temperature"),
        max_tokens=model_data.get("max_tokens"),
        thinking_type=model_data.get("thinking_type"),
        timeout_seconds=model_data.get("timeout_seconds"),
        response_format=model_data.get("response_format"),
        enabled=model_data.get("enabled"),
        network_policy=str(profile_data["network_policy"]),
    )


def redact_runtime_config(config: ResolvedModelConfig) -> dict[str, object]:
    """返回不包含 API key 值的运行配置摘要。"""

    return {
        "role": config.role,
        "profile": config.profile,
        "model_id": config.model_id,
        "provider_id": config.provider_id,
        "provider_type": config.provider_type,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "thinking_type": config.thinking_type,
        "timeout_seconds": config.timeout_seconds,
        "response_format": (
            dict(config.response_format) if config.response_format is not None else None
        ),
        "enabled": config.enabled,
        "network_policy": config.network_policy,
    }


def require_api_key_env(config: ResolvedModelConfig, environ: Mapping[str, str]) -> None:
    """外部 provider 调用前确认 key 环境变量存在，不读取其值到配置对象。"""

    if config.provider_type == "in_process":
        return
    if not config.api_key_env:
        raise RuntimeConfigError(
            "RUNTIME_CONFIG_API_KEY_ENV_MISSING", config.provider_id
        )
    if not environ.get(config.api_key_env, "").strip():
        raise RuntimeConfigError("RUNTIME_CONFIG_API_KEY_MISSING", config.api_key_env)
