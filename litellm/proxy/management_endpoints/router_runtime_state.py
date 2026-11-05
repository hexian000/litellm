from __future__ import annotations

import hashlib
import inspect
import os
import socket
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, JsonValue, StrictInt, TypeAdapter, ValidationError

import litellm
from litellm._version import version
from litellm.router import Router
from litellm.router_utils.pattern_match_deployments import PatternMatchRouter
from litellm.types.router import Deployment

PROTOCOL = "litellm.runtime-state/v1"
BOOT_ID = uuid.uuid4().hex
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
VISIBLE_PARAMETERS = frozenset(
    (
        "model",
        "use_in_pass_through",
        "use_litellm_proxy",
        "use_xai_oauth",
        "merge_reasoning_content_in_choices",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "max_new_tokens",
        "additional_drop_params",
        "allowed_openai_params",
        "drop_params",
        "modify_params",
        "timeout",
        "stream_timeout",
        "max_retries",
        "num_retries",
        "rpm",
        "tpm",
        "weight",
        "order",
    )
)
PARAMETERS = TypeAdapter(Mapping[str, JsonValue])
FALLBACKS: TypeAdapter[Sequence[Mapping[str, Sequence[str]]] | None] = TypeAdapter(
    Sequence[Mapping[str, Sequence[str]]] | None
)
ALIASES = TypeAdapter(Mapping[str, str | Mapping[Literal["model", "hidden"], str | bool]])
SDK_ALIASES = TypeAdapter(Mapping[str, str])
JSON_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class RuntimeRoutingIndices(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_list: tuple[object, ...]
    model_names: frozenset[str]
    model_id_to_deployment_index_map: Mapping[str, StrictInt]
    model_name_to_deployment_indices: Mapping[str, tuple[StrictInt, ...]]


class RuntimeParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    keys: tuple[str, ...]
    values: Mapping[str, JsonValue]
    opaque_keys: tuple[str, ...]


class RuntimeDeployment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    model_name: str
    litellm_params: RuntimeParameters


class RouterRuntimeState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["litellm.runtime-state/v1"] = PROTOCOL
    boot_id: str
    process_id: int
    hostname: str
    version: str
    protocol_source_sha256: str
    default_litellm_params: RuntimeParameters
    sdk_settings: RuntimeParameters
    deployments: tuple[RuntimeDeployment, ...]
    removal_semantics: Literal["reload-or-recreate/v1"] = "reload-or-recreate/v1"


class RouterRuntimeRoutingState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["litellm.runtime-routing-state/v1"] = "litellm.runtime-routing-state/v1"
    runtime: RouterRuntimeState
    routing_sources: RuntimeParameters
    model_names: tuple[str, ...]
    model_name_to_deployment_indices: Mapping[str, tuple[int, ...]]
    opaque_sources: tuple[str, ...]
    selection_semantics: Literal["router-selection/v1"] = "router-selection/v1"
    default_fallbacks_source: Literal["fallbacks[*]"] = "fallbacks[*]"


def runtime_parameters(parameters: object) -> RuntimeParameters:
    if not isinstance(parameters, Mapping) or not all(isinstance(key, str) for key in parameters):
        raise HTTPException(status_code=503, detail="Runtime parameters are not inspectable")
    try:
        return RuntimeParameters(
            keys=tuple(sorted(parameters)),
            values=PARAMETERS.validate_python(
                MappingProxyType({key: value for key, value in parameters.items() if key in VISIBLE_PARAMETERS})
            ),
            opaque_keys=tuple(sorted(key for key in parameters if key not in VISIBLE_PARAMETERS)),
        )
    except ValidationError:
        raise HTTPException(status_code=503, detail="Runtime parameters are not inspectable") from None


def serving_deployment(router: Router, model_id: str) -> RuntimeDeployment:
    index = router.model_id_to_deployment_index_map.get(model_id)
    if index is None:
        raise HTTPException(status_code=409, detail="Selected deployment is not serving in this process")
    raw = router.model_list[index]
    record = raw.model_dump(exclude_unset=True) if isinstance(raw, Deployment) else raw
    if not isinstance(record, dict):
        raise HTTPException(status_code=503, detail="Serving deployment is not inspectable")
    name = record.get("model_name")
    info = record.get("model_info")
    if not isinstance(name, str) or not isinstance(info, dict) or info.get("id") != model_id:
        raise HTTPException(status_code=503, detail="Serving deployment identity is inconsistent")
    return RuntimeDeployment(
        model_id=model_id,
        model_name=name,
        litellm_params=runtime_parameters(record.get("litellm_params")),
    )


def snapshot_runtime(router: Router, model_ids: tuple[str, ...]) -> RouterRuntimeState:
    if len(frozenset(model_ids)) != len(model_ids):
        raise HTTPException(status_code=422, detail="Duplicate deployment selectors")
    return RouterRuntimeState(
        boot_id=BOOT_ID,
        process_id=os.getpid(),
        hostname=socket.gethostname(),
        version=version,
        protocol_source_sha256=SOURCE_SHA256,
        default_litellm_params=runtime_parameters(router.default_litellm_params),
        sdk_settings=runtime_parameters(
            MappingProxyType({"drop_params": litellm.drop_params, "modify_params": litellm.modify_params})
        ),
        deployments=tuple(serving_deployment(router, model_id) for model_id in model_ids),
    )


def routing_value(name: str, value: object) -> JsonValue:
    try:
        if name in ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks"):
            fallbacks = FALLBACKS.validate_python(value, strict=True)
            if fallbacks is not None and any(len(item) != 1 for item in fallbacks):
                raise HTTPException(status_code=503, detail="Runtime routing source is not inspectable")
            return JSON_VALUE.validate_python(fallbacks)
        if name == "model_alias_map":
            return JSON_VALUE.validate_python(SDK_ALIASES.validate_python(value, strict=True))
        aliases = ALIASES.validate_python(value, strict=True)
        if any(
            isinstance(item, Mapping)
            and (not isinstance(item.get("model"), str) or not isinstance(item.get("hidden", False), bool))
            for item in aliases.values()
        ):
            raise HTTPException(status_code=503, detail="Runtime routing source is not inspectable")
        return JSON_VALUE.validate_python(aliases)
    except ValidationError:
        raise HTTPException(status_code=503, detail="Runtime routing source is not inspectable") from None


def object_attributes(value: object) -> Mapping[str, object]:
    return vars(value)


def snapshot_routing(router: Router) -> RouterRuntimeRoutingState:
    if type(router) is not Router:
        raise HTTPException(status_code=503, detail="Custom Router implementation is not inspectable")
    attributes = object_attributes(router)
    sdk_attributes = object_attributes(litellm)
    methods = object_attributes(Router)
    pattern_attributes = object_attributes(router.pattern_router)
    try:
        indices = RuntimeRoutingIndices.model_validate(attributes)
    except ValidationError:
        raise HTTPException(status_code=503, detail="Serving inventory is not inspectable") from None
    if len(indices.model_list) > 10000:
        raise HTTPException(status_code=503, detail="Serving inventory exceeds the inspection bound")
    inventory = tuple(
        serving_deployment(router, model_id)
        for model_id, index in indices.model_id_to_deployment_index_map.items()
        if index in range(len(indices.model_list))
    )
    if (
        len(inventory) != len(indices.model_list)
        or tuple(indices.model_id_to_deployment_index_map.values()) != tuple(range(len(indices.model_list)))
        or indices.model_names != frozenset(entry.model_name for entry in inventory)
        or frozenset(indices.model_name_to_deployment_indices) != indices.model_names
        or any(
            group_indices != tuple(index for index, entry in enumerate(inventory) if entry.model_name == name)
            for name, group_indices in indices.model_name_to_deployment_indices.items()
        )
    ):
        raise HTTPException(status_code=503, detail="Serving inventory indices are inconsistent")
    sources = MappingProxyType(
        {
            "fallbacks": routing_value("fallbacks", attributes["fallbacks"]),
            "context_window_fallbacks": routing_value(
                "context_window_fallbacks", attributes["context_window_fallbacks"]
            ),
            "content_policy_fallbacks": routing_value(
                "content_policy_fallbacks", attributes["content_policy_fallbacks"]
            ),
            "model_group_alias": routing_value("model_group_alias", router.model_group_alias),
            "model_alias_map": routing_value("model_alias_map", litellm.model_alias_map),
        }
    )
    extensions = MappingProxyType(
        {
            "default_deployment": attributes["default_deployment"] is not None,
            "provider_default_deployment_ids": router.provider_default_deployment_ids,
            "pattern_router": type(router.pattern_router) is not PatternMatchRouter
            or any(name != "patterns" or bool(value) for name, value in pattern_attributes.items()),
            "team_pattern_routers": router.team_pattern_routers,
            "team_model_to_deployment_indices": router.team_model_to_deployment_indices,
            "auto_routers": router.auto_routers,
            "complexity_routers": router.complexity_routers,
            "adaptive_routers": router.adaptive_routers,
            "quality_routers": router.quality_routers,
            "routing_plugins": router.routing_plugins,
            "optional_callbacks": attributes.get("optional_callbacks"),
            "callbacks": sdk_attributes["callbacks"],
            "input_callback": sdk_attributes["input_callback"],
            "async_input_callback": sdk_attributes["_async_input_callback"],
            "custom_provider_map": sdk_attributes.get("custom_provider_map"),
            "instance_method_overrides": any(
                name in attributes for name, value in methods.items() if inspect.isfunction(value)
            ),
            "routing_strategy": router.routing_strategy != "simple-shuffle",
            "routing_groups": attributes["_routing_groups"],
        }
    )
    return RouterRuntimeRoutingState(
        runtime=snapshot_runtime(router, tuple(entry.model_id for entry in inventory)),
        routing_sources=RuntimeParameters(keys=tuple(sorted(sources)), values=sources, opaque_keys=()),
        model_names=tuple(sorted(indices.model_names)),
        model_name_to_deployment_indices=MappingProxyType(
            {name: group_indices for name, group_indices in indices.model_name_to_deployment_indices.items()}
        ),
        opaque_sources=tuple(sorted(name for name, value in extensions.items() if value)),
    )
