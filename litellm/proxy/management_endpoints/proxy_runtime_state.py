from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from fastapi import HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
)

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.management_endpoints.router_runtime_state import (
    BOOT_ID,
    JSON_VALUE,
    SDK_ALIASES,
    RouterRuntimeRoutingState,
    RuntimeParameters,
    snapshot_routing,
)
from litellm.router import Router

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OBJECT_MAPPING = TypeAdapter(Mapping[str, object])
EMPTY_PARAMETERS: Mapping[str, object] = MappingProxyType({})
CALLBACK_REGISTRIES = (
    "callbacks",
    "input_callback",
    "success_callback",
    "failure_callback",
    "_async_input_callback",
    "_async_success_callback",
    "_async_failure_callback",
)


class LoadedProxyInputs(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, allow_inf_nan=False)

    user_model: StrictStr | None
    user_max_tokens: StrictInt | None
    user_temperature: StrictFloat | None
    user_request_timeout: StrictFloat | None
    user_api_base: StrictStr | None
    general_settings: Mapping[str, object]


class ProxyDispatchInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    deployment_names: Sequence[StrictStr]
    team_public_model_names: frozenset[StrictStr]
    pass_through_all_models: StrictBool


class RuntimeAuthentication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    credential_fingerprint: str
    aliases: Mapping[str, str] | None
    router_settings: RuntimeParameters
    scope_semantics: Literal["same-credential-no-team-or-project/v1"] = "same-credential-no-team-or-project/v1"


class ProxyRuntimePreprocessingInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_source_sha256: str
    routing: RouterRuntimeRoutingState
    proxy_inputs: RuntimeParameters
    dispatch_inputs: ProxyDispatchInputs
    general_settings: RuntimeParameters
    agentic_cli_drop_params_configured: bool
    opaque_sources: tuple[str, ...]
    request_sources: tuple[Literal["body", "headers", "query", "route"], ...] = ("body", "headers", "query", "route")
    acceptance_semantics: Literal["refuse-unreviewed-sources/v1"] = "refuse-unreviewed-sources/v1"


class ProxyRuntimePreprocessingState(ProxyRuntimePreprocessingInputs):
    protocol: Literal["litellm.runtime-preprocessing-state/v1"] = "litellm.runtime-preprocessing-state/v1"
    authentication: RuntimeAuthentication


def inspect_mapping(value: object) -> Mapping[str, object]:
    if type(value) is not dict:
        raise HTTPException(status_code=503, detail="Proxy preprocessing source is not inspectable")
    try:
        return OBJECT_MAPPING.validate_python(value, strict=True)
    except ValidationError:
        raise HTTPException(status_code=503, detail="Proxy preprocessing source is not inspectable") from None


def projected_parameters(parameters: Mapping[str, object], visible: frozenset[str]) -> RuntimeParameters:
    try:
        return RuntimeParameters(
            keys=tuple(sorted(parameters)),
            values=MappingProxyType(
                {
                    key: JSON_VALUE.validate_python(value, strict=True)
                    for key, value in parameters.items()
                    if key in visible
                }
            ),
            opaque_keys=tuple(sorted(key for key in parameters if key not in visible)),
        )
    except ValidationError:
        raise HTTPException(status_code=503, detail="Proxy preprocessing source is not inspectable") from None


def empty_authentication_extension(value: object) -> bool:
    return value is None or type(value) is dict and not value


def active_callback_registry(value: object) -> bool:
    if type(value) not in (list, tuple):
        raise HTTPException(status_code=503, detail="Callback registry is not inspectable")
    return bool(value)


def snapshot_authentication(
    authentication: UserAPIKeyAuth,
    credential: tuple[tuple[str, ...], ...],
) -> RuntimeAuthentication:
    if not credential:
        raise HTTPException(status_code=503, detail="Authentication credential is not inspectable")
    if type(authentication) is not UserAPIKeyAuth:
        raise HTTPException(status_code=503, detail="Custom authentication context is not inspectable")
    auth_attributes = inspect_mapping(vars(authentication))
    if any(
        auth_attributes.get(name) is not None
        for name in ("team_id", "project_id", "organization_id", "org_id", "end_user_id")
    ):
        raise HTTPException(status_code=503, detail="Hierarchical authentication context is not inspectable")
    if any(
        not empty_authentication_extension(auth_attributes.get(name))
        for name in (
            "team_model_aliases",
            "metadata",
            "team_metadata",
            "project_metadata",
            "organization_metadata",
            "allowed_model_region",
            "jwt_claims",
            "config",
            "permissions",
            "budget_fallbacks",
            "object_permission",
            "team_object_permission",
            "end_user_object_permission",
        )
    ):
        raise HTTPException(status_code=503, detail="Authentication extensions are not inspectable")
    try:
        aliases_value = auth_attributes.get("aliases")
        aliases = None if aliases_value is None else SDK_ALIASES.validate_python(aliases_value, strict=True)
    except ValidationError:
        raise HTTPException(status_code=503, detail="Proxy preprocessing source is not inspectable") from None
    key_settings_value = auth_attributes.get("router_settings")
    key_settings = inspect_mapping(key_settings_value) if key_settings_value is not None else EMPTY_PARAMETERS
    return RuntimeAuthentication(
        credential_fingerprint=hashlib.sha256(
            json.dumps((BOOT_ID, credential), separators=(",", ":")).encode()
        ).hexdigest(),
        aliases=aliases,
        router_settings=projected_parameters(
            key_settings,
            frozenset(
                (
                    "fallbacks",
                    "context_window_fallbacks",
                    "content_policy_fallbacks",
                    "num_retries",
                    "timeout",
                    "routing_strategy",
                    "enable_tag_filtering",
                    "model_group_retry_policy",
                )
            ),
        ),
    )


def snapshot_preprocessing_inputs(
    router: Router,
    proxy_attributes: Mapping[str, object],
    config_attributes: Mapping[str, object],
) -> ProxyRuntimePreprocessingInputs:
    router_attributes = inspect_mapping(vars(router))
    sdk_attributes = inspect_mapping(vars(litellm))
    try:
        inputs = LoadedProxyInputs.model_validate(proxy_attributes)
        dispatch_inputs = ProxyDispatchInputs.model_validate(
            MappingProxyType(
                {
                    "deployment_names": router_attributes["deployment_names"],
                    "team_public_model_names": router_attributes["team_public_model_names"],
                    "pass_through_all_models": router.router_general_settings.pass_through_all_models,
                }
            )
        )
    except ValidationError:
        raise HTTPException(status_code=503, detail="Proxy preprocessing source is not inspectable") from None
    settings = inspect_mapping(inputs.general_settings)
    if settings.get("completion_model") is not None and not isinstance(settings["completion_model"], str):
        raise HTTPException(status_code=503, detail="Proxy preprocessing source is not inspectable")
    config = inspect_mapping(config_attributes.get("config"))
    sdk_config = inspect_mapping(config["litellm_settings"]) if "litellm_settings" in config else EMPTY_PARAMETERS
    extensions = MappingProxyType(
        {
            **MappingProxyType(
                {f"litellm.{name}": active_callback_registry(sdk_attributes.get(name)) for name in CALLBACK_REGISTRIES}
            ),
            "proxy.user_api_base": inputs.user_api_base is not None,
            "proxy.user_custom_auth": proxy_attributes.get("user_custom_auth") is not None,
            "proxy.user_headers": bool(proxy_attributes.get("user_headers")),
            "proxy.user_config": bool(proxy_attributes.get("user_config")),
            "sdk.enable_model_config_credential_overrides": litellm.enable_model_config_credential_overrides,
        }
    )
    return ProxyRuntimePreprocessingInputs(
        protocol_source_sha256=SOURCE_SHA256,
        routing=snapshot_routing(router),
        dispatch_inputs=dispatch_inputs,
        proxy_inputs=projected_parameters(
            MappingProxyType(
                {
                    "user_model": inputs.user_model,
                    "user_max_tokens": inputs.user_max_tokens,
                    "user_temperature": inputs.user_temperature,
                    "user_request_timeout": inputs.user_request_timeout,
                    "user_api_base": inputs.user_api_base,
                }
            ),
            frozenset(("user_model", "user_max_tokens", "user_temperature", "user_request_timeout")),
        ),
        general_settings=projected_parameters(settings, frozenset(("completion_model",))),
        agentic_cli_drop_params_configured="drop_params" in sdk_config,
        opaque_sources=tuple(sorted(name for name, active in extensions.items() if active)),
    )


def snapshot_preprocessing(
    router: Router,
    proxy_attributes: Mapping[str, object],
    config_attributes: Mapping[str, object],
    authentication: UserAPIKeyAuth,
    credential: tuple[tuple[str, ...], ...],
) -> ProxyRuntimePreprocessingState:
    auth = snapshot_authentication(authentication, credential)
    inputs = snapshot_preprocessing_inputs(router, proxy_attributes, config_attributes)
    return ProxyRuntimePreprocessingState(
        protocol_source_sha256=inputs.protocol_source_sha256,
        routing=inputs.routing,
        proxy_inputs=inputs.proxy_inputs,
        dispatch_inputs=inputs.dispatch_inputs,
        general_settings=inputs.general_settings,
        authentication=auth,
        agentic_cli_drop_params_configured=inputs.agentic_cli_drop_params_configured,
        opaque_sources=inputs.opaque_sources,
    )
