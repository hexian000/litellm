import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.proxy.management_endpoints import router_runtime_state
from litellm.proxy.management_endpoints.router_settings_endpoints import router
from litellm.proxy.management_endpoints.model_management_endpoints import update_db_model
from litellm.router import Router
from litellm.types.router import Deployment, updateDeployment


def runtime_router():
    return Router(
        model_list=[
            {
                "model_name": "bounded",
                "model_info": {"id": "selected"},
                "litellm_params": {
                    "model": "deepseek/deepseek-chat",
                    "api_key": "private-credential",
                    "max_new_tokens": 65537,
                    "unknown_retained": "private-provider-field",
                },
            },
            {
                "model_name": "unrelated",
                "model_info": {"id": "unrelated"},
                "litellm_params": {"model": "deepseek/deepseek-chat", "api_key": "other-private"},
            },
        ],
        default_litellm_params={"additional_drop_params": ["max_tokens"]},
    )


def runtime_client(serving_router, role=LitellmUserRoles.PROXY_ADMIN):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role)
    return TestClient(app)


def test_runtime_http_preserves_effective_defaults_and_retained_keyspace():
    serving = runtime_router()
    serving.update_settings(default_litellm_params={})
    client = runtime_client(serving)
    config = AsyncMock(return_value={"router_settings": {"default_litellm_params": {}}})
    with (
        patch.object(proxy_server, "llm_router", serving),
        patch.object(proxy_server.proxy_config, "get_config", config),
    ):
        legacy = client.get("/router/settings").json()
        response = client.get("/router/runtime-state", params={"model_id": "selected"})
        again = client.get("/router/runtime-state", params={"model_id": "selected"})
    assert legacy["current_values"]["default_litellm_params"] == {}
    assert response.status_code == 200
    body = response.json()
    assert body == again.json()
    assert body["protocol"] == "litellm.runtime-state/v1"
    assert (
        body["protocol_source_sha256"] == hashlib.sha256(Path(router_runtime_state.__file__).read_bytes()).hexdigest()
    )
    assert body["boot_id"] and body["process_id"] > 0 and body["hostname"]
    assert body["default_litellm_params"]["values"]["additional_drop_params"] == ["max_tokens"]
    deployment = body["deployments"][0]
    assert len(body["deployments"]) == 1
    assert deployment["model_id"] == "selected" and deployment["model_name"] == "bounded"
    assert deployment["litellm_params"]["values"]["max_new_tokens"] == 65537
    assert "unknown_retained" in deployment["litellm_params"]["keys"]
    assert "unknown_retained" in deployment["litellm_params"]["opaque_keys"]
    assert "api_key" in deployment["litellm_params"]["opaque_keys"]
    assert "private" not in response.text
    assert body["removal_semantics"] == "reload-or-recreate/v1"
    assert serving.get_deployment("unrelated") is not None


@pytest.mark.parametrize("patch_value, expected", [(None, 65537), ("", ""), (65536, 65536)])
def test_runtime_http_reports_actual_patch_merge(patch_value, expected):
    original = Deployment(
        model_name="bounded",
        litellm_params={"model": "deepseek/deepseek-chat", "max_new_tokens": 65537},
        model_info={"id": "selected"},
    )
    with patch.dict("os.environ", {"LITELLM_SALT_KEY": "test-only-not-a-service-key"}):
        merged = update_db_model(original, updateDeployment(litellm_params={"max_new_tokens": patch_value}))
        parameters = {
            key: decrypt_value_helper(value, key=key, return_original_value=True)
            for key, value in json.loads(merged["litellm_params"]).items()
        }
    serving = Router(
        model_list=[
            {
                "model_name": "bounded",
                "model_info": {"id": "selected"},
                "litellm_params": parameters,
            }
        ]
    )
    with patch.object(proxy_server, "llm_router", serving):
        response = runtime_client(serving).get("/router/runtime-state", params={"model_id": "selected"})
    assert response.status_code == 200
    assert response.json()["deployments"][0]["litellm_params"]["values"]["max_new_tokens"] == expected


def test_runtime_http_reports_omitted_patch_retention():
    original = Deployment(
        model_name="bounded",
        litellm_params={"model": "deepseek/deepseek-chat", "max_new_tokens": 65537},
        model_info={"id": "selected"},
    )
    merged = update_db_model(original, updateDeployment(litellm_params={"max_tokens": 65536}))
    assert json.loads(merged["litellm_params"])["max_new_tokens"] == 65537


@pytest.mark.parametrize("role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
def test_runtime_http_refuses_non_admin(role):
    with patch.object(proxy_server, "llm_router", runtime_router()):
        response = runtime_client(None, role).get("/router/runtime-state", params={"model_id": "selected"})
    assert response.status_code == 403


@pytest.mark.parametrize(
    "selectors, status",
    [
        ([], 422),
        (["missing"], 409),
        (["selected", "missing"], 409),
        (["selected", "selected"], 422),
        (["selected"] * 101, 422),
    ],
)
def test_runtime_http_refuses_ambiguous_or_missing_selectors(selectors, status):
    serving = runtime_router()
    with patch.object(proxy_server, "llm_router", serving):
        response = runtime_client(serving).get(
            "/router/runtime-state", params=[("model_id", item) for item in selectors]
        )
    assert response.status_code == status
    assert "deployments" not in response.json()


def test_runtime_http_refuses_uninitialized_router():
    with patch.object(proxy_server, "llm_router", None):
        response = runtime_client(None).get("/router/runtime-state", params={"model_id": "selected"})
    assert response.status_code == 503


def test_runtime_projection_retains_null_unknown_key_without_disclosing_value():
    state = router_runtime_state.runtime_parameters({"unknown_null": None, "unknown_empty": "", "max_tokens": None})
    assert state.keys == ("max_tokens", "unknown_empty", "unknown_null")
    assert state.opaque_keys == ("unknown_empty", "unknown_null")
    assert state.values == {"max_tokens": None}


def test_snapshot_observes_the_sdk_drop_that_defeats_an_ingress_cap():
    from litellm.utils import get_optional_params

    serving = runtime_router()
    arguments = {"max_tokens": 65536}
    serving._update_kwargs_with_default_litellm_params(arguments)
    optional = get_optional_params(
        model="deepseek-chat",
        custom_llm_provider="deepseek",
        max_tokens=arguments["max_tokens"],
        additional_drop_params=arguments["additional_drop_params"],
    )
    assert "max_tokens" not in optional
    snapshot = router_runtime_state.snapshot_runtime(serving, ("selected",))
    assert snapshot.default_litellm_params.values["additional_drop_params"] == ["max_tokens"]


def test_snapshot_observes_current_entry_without_dropping_unknown_null():
    serving = runtime_router()
    serving.model_list[0]["litellm_params"]["unknown_null"] = None
    serving.model_list[0]["litellm_params"]["max_new_tokens"] = 65538
    snapshot = router_runtime_state.snapshot_runtime(serving, ("selected",))
    assert "unknown_null" in snapshot.deployments[0].litellm_params.opaque_keys
    assert snapshot.deployments[0].litellm_params.values["max_new_tokens"] == 65538


def test_snapshot_refuses_inconsistent_serving_identity():
    from fastapi import HTTPException

    serving = runtime_router()
    serving.model_list[0]["model_info"]["id"] = "different"
    with pytest.raises(HTTPException) as error:
        router_runtime_state.snapshot_runtime(serving, ("selected",))
    assert error.value.status_code == 503
