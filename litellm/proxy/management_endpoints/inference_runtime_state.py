from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeVar

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, StrictStr, TypeAdapter, ValidationError
from starlette.requests import Request
from starlette.types import Message

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import get_team_object
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.proxy_runtime_state import (
    EMPTY_PARAMETERS,
    ProxyRuntimePreprocessingInputs,
    empty_authentication_extension,
    inspect_mapping,
    projected_parameters,
    snapshot_preprocessing_inputs,
)
from litellm.proxy.management_endpoints.router_runtime_state import BOOT_ID, SDK_ALIASES, RuntimeParameters
from litellm.proxy.spend_tracking.budget_reservation import release_budget_reservation

EMPTY_STATE: MappingProxyType[str, object] = MappingProxyType({})
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OPTIONAL_STRING: TypeAdapter[str | None] = TypeAdapter(StrictStr | None)
REQUEST_BODY = TypeAdapter(Mapping[str, object])
PARSED_BODY = TypeAdapter(tuple[tuple[str, ...], Mapping[str, object]])
INFERENCE_PATH = Literal["/chat/completions", "/v1/chat/completions", "/responses", "/v1/responses", "/v1/messages"]
VISIBLE_ROUTER_SETTINGS = frozenset(
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
)
AUTHENTICATION_EXTENSIONS = (
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
AUTHENTICATION_HEADERS = frozenset(
    ("authorization", "x-litellm-api-key", "x-api-key", "api-key", "x-goog-api-key", "x-litellm-inspection-credential")
)
INSPECTION_TASK_RESULT = TypeVar("INSPECTION_TASK_RESULT")


class InferenceInspectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: INFERENCE_PATH
    body: StrictStr
    headers: tuple[tuple[StrictStr, StrictStr], ...] = ()
    query: StrictStr = ""


class RuntimeInferenceAuthentication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scope_semantics: Literal["resolved-inference-credential/v1"] = "resolved-inference-credential/v1"
    credential_fingerprint: str
    request_fingerprint: str
    scopes: Mapping[str, StrictStr | None]
    aliases: Mapping[str, str] | None
    team_model_aliases: Mapping[str, str] | None
    key_router_settings: RuntimeParameters
    effective_router_settings: RuntimeParameters
    metadata: Mapping[str, RuntimeParameters]


class InferenceRuntimePreprocessingState(ProxyRuntimePreprocessingInputs):
    protocol: Literal["litellm.runtime-inference-preprocessing-state/v1"] = (
        "litellm.runtime-inference-preprocessing-state/v1"
    )
    inference_source_sha256: str
    authentication: RuntimeInferenceAuthentication
    authenticated_request: RuntimeParameters


@dataclass(frozen=True, slots=True)
class InspectionBody:
    body: bytes

    async def receive(self) -> Message:
        return MappingProxyType({"type": "http.request", "body": self.body, "more_body": False}).copy()


def inspected_aliases(value: object) -> Mapping[str, str] | None:
    return None if value is None else SDK_ALIASES.validate_python(value, strict=True)


def inspection_request(envelope: InferenceInspectionRequest, credential: str, original: Request) -> Request:
    if (
        not credential.startswith("Bearer sk-")
        or credential.count(" ") != 1
        or "\r" in credential
        or "\n" in credential
    ):
        raise HTTPException(status_code=400, detail="Inspection requires one original virtual-key bearer")
    if any(
        name.lower() in AUTHENTICATION_HEADERS
        or name.lower() == "content-length"
        or not name.isascii()
        or not value.isascii()
        or "\r" in name + value
        or "\n" in name + value
        for name, value in envelope.headers
    ):
        raise HTTPException(status_code=400, detail="Inspection request headers are not supported")
    if (
        len(frozenset(name.lower() for name, _ in envelope.headers)) != len(envelope.headers)
        or not envelope.query.isascii()
    ):
        raise HTTPException(status_code=400, detail="Inspection request headers or query are ambiguous")
    try:
        _ = REQUEST_BODY.validate_json(envelope.body, strict=True)
    except ValidationError:
        raise HTTPException(status_code=400, detail="Inspection body must contain a JSON object") from None
    body = envelope.body.encode()
    original_scope = inspect_mapping(original.scope)
    return Request(
        MappingProxyType(
            {
                "type": "http",
                "http_version": original_scope.get("http_version", "1.1"),
                "scheme": original_scope.get("scheme", "http"),
                "server": original_scope.get("server"),
                "client": original_scope.get("client"),
                "app": original_scope.get("app"),
                "method": "POST",
                "path": envelope.path,
                "raw_path": envelope.path.encode(),
                "root_path": "",
                "query_string": envelope.query.encode(),
                "headers": tuple((name.lower().encode(), value.encode()) for name, value in envelope.headers)
                + ((b"authorization", credential.encode()), (b"content-length", str(len(body)).encode())),
                "path_params": EMPTY_STATE.copy(),
                "state": EMPTY_STATE.copy(),
                "route": None,
            }
        ).copy(),
        receive=InspectionBody(body).receive,
    )


def authentication_parameters(value: object) -> RuntimeParameters:
    parameters = EMPTY_PARAMETERS if value is None else inspect_mapping(value)
    return projected_parameters(parameters, VISIBLE_ROUTER_SETTINGS)


def resolved_authentication(
    authentication: UserAPIKeyAuth,
    envelope: InferenceInspectionRequest,
    credential: str,
    effective_router_settings: object,
) -> RuntimeInferenceAuthentication:
    if type(authentication) is not UserAPIKeyAuth:
        raise HTTPException(status_code=503, detail="Custom authentication context is not inspectable")
    attributes = inspect_mapping(vars(authentication))
    if any(attributes.get(name) is not None for name in ("organization_id", "org_id", "end_user_id")) or any(
        not empty_authentication_extension(attributes.get(name)) for name in AUTHENTICATION_EXTENSIONS
    ):
        raise HTTPException(status_code=503, detail="Authentication extensions are not inspectable")
    try:
        return RuntimeInferenceAuthentication(
            credential_fingerprint=hashlib.sha256(
                json.dumps((BOOT_ID, ((credential,), (), ())), separators=(",", ":")).encode()
            ).hexdigest(),
            request_fingerprint=hashlib.sha256(
                json.dumps((BOOT_ID, envelope.model_dump()), separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest(),
            scopes=MappingProxyType(
                {
                    name: OPTIONAL_STRING.validate_python(attributes.get(name), strict=True)
                    for name in ("team_id", "project_id", "user_id")
                }
            ),
            aliases=inspected_aliases(attributes.get("aliases")),
            team_model_aliases=inspected_aliases(attributes.get("team_model_aliases")),
            key_router_settings=authentication_parameters(attributes.get("router_settings")),
            effective_router_settings=authentication_parameters(effective_router_settings),
            metadata=MappingProxyType(
                {
                    name: projected_parameters(
                        EMPTY_PARAMETERS if attributes.get(name) is None else inspect_mapping(attributes[name]),
                        frozenset(),
                    )
                    for name in ("metadata", "team_metadata", "project_metadata")
                }
            ),
        )
    except ValidationError:
        raise HTTPException(status_code=503, detail="Authentication context is not inspectable") from None


async def finish_inspection_task(task: asyncio.Task[INSPECTION_TASK_RESULT]) -> INSPECTION_TASK_RESULT:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
    return task.result()


async def inspect_inference_preprocessing(
    envelope: InferenceInspectionRequest, credential: str, original: Request
) -> InferenceRuntimePreprocessingState:
    from litellm.proxy import proxy_server as proxy

    if proxy.llm_router is None:
        raise HTTPException(status_code=503, detail="Router is not initialized")
    proxy_attributes = inspect_mapping(vars(proxy))
    if (
        type(proxy.proxy_config) is not proxy.ProxyConfig
        or type(proxy.proxy_logging_obj) is not proxy.ProxyLogging
        or proxy_attributes.get("user_custom_auth") is not None
        or inspect_mapping(proxy_attributes.get("general_settings")).get("litellm_key_header_name")
    ):
        raise HTTPException(status_code=503, detail="Custom proxy authentication or preprocessing is not inspectable")
    if any(
        name in vars(instance)
        for instance in (proxy.proxy_config, proxy.proxy_logging_obj)
        for name, value in REQUEST_BODY.validate_python(vars(type(instance)).copy(), strict=True).items()
        if inspect.isfunction(value) or isinstance(value, (staticmethod, classmethod))
    ):
        raise HTTPException(status_code=503, detail="Custom proxy preprocessing is not inspectable")
    inference_request = inspection_request(envelope, credential, original)
    authentication_task = asyncio.create_task(
        user_api_key_auth(
            request=inference_request,
            api_key=credential,
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            custom_litellm_key_header=None,
        )
    )
    try:
        authentication = await asyncio.shield(authentication_task)
    except asyncio.CancelledError as cancellation:
        try:
            cancelled_authentication = await finish_inspection_task(authentication_task)
        except (HTTPException, ProxyException):
            raise cancellation from None
        await finish_inspection_task(
            asyncio.create_task(release_budget_reservation(cancelled_authentication.budget_reservation))
        )
        raise
    except (HTTPException, ProxyException):
        raise HTTPException(
            status_code=403, detail="Inference credential is not authorized for the inspected request"
        ) from None
    try:
        if authentication.team_id is not None:
            try:
                await get_team_object(
                    team_id=authentication.team_id,
                    prisma_client=proxy.prisma_client,
                    user_api_key_cache=proxy.user_api_key_cache,
                    proxy_logging_obj=proxy.proxy_logging_obj,
                )
            except HTTPException:
                raise HTTPException(status_code=503, detail="Inference team state is not inspectable") from None
        effective_settings = await proxy.proxy_config.get_hierarchical_router_settings(
            user_api_key_dict=authentication,
            prisma_client=proxy.prisma_client,
            proxy_logging_obj=proxy.proxy_logging_obj,
        )
        auth = resolved_authentication(authentication, envelope, credential, effective_settings)
        inputs = snapshot_preprocessing_inputs(proxy.llm_router, vars(proxy), vars(proxy.proxy_config))
        return InferenceRuntimePreprocessingState(
            protocol_source_sha256=inputs.protocol_source_sha256,
            inference_source_sha256=SOURCE_SHA256,
            routing=inputs.routing,
            proxy_inputs=inputs.proxy_inputs,
            dispatch_inputs=inputs.dispatch_inputs,
            general_settings=inputs.general_settings,
            authentication=auth,
            authenticated_request=projected_parameters(
                PARSED_BODY.validate_python(inference_request.scope.get("parsed_body"), strict=True)[1],
                frozenset(("model", "max_tokens", "max_completion_tokens", "max_output_tokens")),
            ),
            agentic_cli_drop_params_configured=inputs.agentic_cli_drop_params_configured,
            opaque_sources=inputs.opaque_sources,
        )
    finally:
        release = asyncio.create_task(release_budget_reservation(authentication.budget_reservation))
        try:
            await asyncio.shield(release)
        except asyncio.CancelledError:
            await finish_inspection_task(release)
            raise
