from __future__ import annotations

import hashlib
import os
import socket
import uuid
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

import litellm
from litellm._version import version
from litellm.router import Router
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
