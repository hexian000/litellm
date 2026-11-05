import json
from contextlib import ExitStack
from unittest.mock import patch

import litellm
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from litellm.proxy import proxy_server as proxy
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.management_endpoints.proxy_runtime_state import CALLBACK_REGISTRIES
from litellm.proxy.management_endpoints.router_settings_endpoints import router
from litellm.router import Router


@pytest.fixture
def serving():
    instance = Router(
        model_list=[
            {
                "model_name": "bounded",
                "model_info": {"id": "selected"},
                "litellm_params": {"model": "deepseek/deepseek-chat", "api_key": "private-provider-credential"},
            }
        ]
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(proxy, "llm_router", instance))
        for name in CALLBACK_REGISTRIES:
            stack.enter_context(patch.object(litellm, name, []))
        for name in (
            "user_model",
            "user_max_tokens",
            "user_temperature",
            "user_request_timeout",
            "user_api_base",
            "user_custom_auth",
        ):
            stack.enter_context(patch.object(proxy, name, None))
        stack.enter_context(patch.object(proxy, "general_settings", {}))
        stack.enter_context(patch.object(proxy.proxy_config, "config", {}))
        stack.enter_context(patch.object(litellm, "model_alias_map", {}))
        yield instance


def client_for(auth):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    return TestClient(app, headers={"Authorization": "Bearer local-inspection-only"})


async def preprocess(auth, model="bounded"):
    data = {"model": model, "messages": [{"role": "user", "content": "local only"}], "max_tokens": 65536}
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [(b"host", b"127.0.0.1")],
            "server": ("127.0.0.1", 80),
            "client": ("127.0.0.1", 12345),
        }
    )
    request._body = json.dumps(data).encode()
    result, _ = await ProxyBaseLLMRequestProcessing(data=data).common_processing_pre_call_logic(
        request=request,
        general_settings=proxy.general_settings,
        user_api_key_dict=auth,
        proxy_logging_obj=proxy.proxy_logging_obj,
        proxy_config=proxy.proxy_config,
        route_type="acompletion",
        user_model=proxy.user_model,
        user_max_tokens=proxy.user_max_tokens,
        user_temperature=proxy.user_temperature,
        user_request_timeout=proxy.user_request_timeout,
        user_api_base=proxy.user_api_base,
        llm_router=proxy.llm_router,
    )
    return result


@pytest.mark.asyncio
async def test_http_observes_actual_preprocessing_override_that_legacy_graph_misses(serving):
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    client = client_for(auth)
    await preprocess(auth)
    before_graph = client.get("/router/runtime-state/routing").json()
    before = client.get("/router/runtime-state/preprocessing")
    assert before.status_code == 200, before.text
    preserved = await preprocess(auth)
    with patch.object(proxy, "user_max_tokens", 65537):
        replaced = await preprocess(auth)
        after_graph = client.get("/router/runtime-state/routing").json()
        after = client.get("/router/runtime-state/preprocessing")
    assert preserved["max_tokens"] == 65536
    assert replaced["max_tokens"] == 65537
    assert before_graph == after_graph
    assert after.status_code == 200, after.text
    body = after.json()
    assert body["protocol"] == "litellm.runtime-preprocessing-state/v1"
    assert body["proxy_inputs"]["values"]["user_max_tokens"] == 65537
    assert before.json()["proxy_inputs"]["values"]["user_max_tokens"] is None
    assert body["routing"] == after_graph
    assert body["routing"]["runtime"]["boot_id"] == before.json()["routing"]["runtime"]["boot_id"]
    assert "private-provider-credential" not in after.text
    assert "local-inspection-only" not in after.text


@pytest.mark.asyncio
async def test_http_observes_loaded_model_override_and_key_aliases_and_router_settings(serving):
    auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        aliases={"proxy-model": "bounded"},
        router_settings={"fallbacks": [{"bounded": ["other"]}], "timeout": 4},
    )
    with (
        patch.object(proxy, "general_settings", {"completion_model": "proxy-model"}),
        patch.object(proxy, "user_model", "cli-model"),
    ):
        result = await preprocess(auth, model="requested")
        response = client_for(auth).get("/router/runtime-state/preprocessing")
    assert result["model"] == "bounded"
    assert result["router_settings_override"] == auth.router_settings
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["general_settings"]["values"] == {"completion_model": "proxy-model"}
    assert body["proxy_inputs"]["values"]["user_model"] == "cli-model"
    assert body["authentication"]["aliases"] == auth.aliases
    assert body["authentication"]["router_settings"]["values"] == auth.router_settings
    assert body["authentication"]["scope_semantics"] == "same-credential-no-team-or-project/v1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("team_id", "team"),
        ("project_id", "project"),
        ("end_user_id", "user"),
        ("team_model_aliases", {"bounded": "other"}),
        ("metadata", {"callbacks": "private-metadata"}),
        ("team_metadata", {"model_config": "private-credential-config"}),
        ("project_metadata", {"model_config": "private-credential-config"}),
        ("router_settings", "fallbacks: [{bounded: [other]}]"),
        ("aliases", {"bounded": 1}),
    ],
)
def test_http_refuses_uninspectable_authentication_context_without_disclosure(serving, field, value):
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    setattr(auth, field, value)
    response = client_for(auth).get("/router/runtime-state/preprocessing")
    assert response.status_code == 503, response.text
    assert "private" not in response.text


@pytest.mark.parametrize("value", [False, "65537", 65537.5])
def test_http_refuses_coercible_loaded_output_overrides(serving, value):
    with patch.object(proxy, "user_max_tokens", value):
        response = client_for(UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)).get(
            "/router/runtime-state/preprocessing"
        )
    assert response.status_code == 503


def test_http_marks_secret_extensions_and_unknown_parameter_keys_opaque(serving):
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, router_settings={"unknown": "private-value"})
    with (
        patch.object(proxy, "user_api_base", "https://private-provider.example"),
        patch.object(proxy, "general_settings", {"unknown": "private-config"}),
        patch.object(litellm, "_async_input_callback", ["private-callback"]),
    ):
        response = client_for(auth).get("/router/runtime-state/preprocessing")
    assert response.status_code == 200, response.text
    assert response.json()["opaque_sources"] == ["litellm._async_input_callback", "proxy.user_api_base"]
    assert response.json()["general_settings"]["opaque_keys"] == ["unknown"]
    assert response.json()["authentication"]["router_settings"]["opaque_keys"] == ["unknown"]
    assert response.json()["request_sources"] == ["body", "headers", "query", "route"]
    assert "private" not in response.text


def test_http_binds_credential_fingerprint_to_current_credential_and_process(serving):
    client = client_for(UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN))
    first = client.get("/router/runtime-state/preprocessing").json()
    repeated = client.get("/router/runtime-state/preprocessing").json()
    other = client.get("/router/runtime-state/preprocessing", headers={"Authorization": "Bearer different"}).json()
    assert first["authentication"]["credential_fingerprint"] == repeated["authentication"]["credential_fingerprint"]
    assert first["authentication"]["credential_fingerprint"] != other["authentication"]["credential_fingerprint"]


@pytest.mark.parametrize("role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
def test_http_refuses_non_admin(serving, role):
    assert client_for(UserAPIKeyAuth(user_role=role)).get("/router/runtime-state/preprocessing").status_code == 403


def test_full_proxy_actual_master_key_authentication(serving):
    with patch.object(proxy, "master_key", "sk-local-master-inspection-only"):
        client = TestClient(proxy.app)
        denied = client.get("/router/runtime-state/preprocessing")
        accepted = client.get(
            "/router/runtime-state/preprocessing", headers={"Authorization": "Bearer sk-local-master-inspection-only"}
        )
    assert denied.status_code in (401, 403)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["routing"]["runtime"]["deployments"][0]["model_id"] == "selected"


def test_http_observes_proxy_dispatch_inputs_and_cli_drop_presence(serving):
    with patch.object(proxy.proxy_config, "config", {"litellm_settings": {"drop_params": False}}):
        serving.router_general_settings.pass_through_all_models = True
        serving.team_public_model_names = frozenset(("public-team-model",))
        response = client_for(UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)).get(
            "/router/runtime-state/preprocessing"
        )
    assert response.status_code == 200, response.text
    assert response.json()["dispatch_inputs"] == {
        "deployment_names": ["deepseek/deepseek-chat"],
        "team_public_model_names": ["public-team-model"],
        "pass_through_all_models": True,
    }
    assert response.json()["agentic_cli_drop_params_configured"] is True


@pytest.mark.parametrize(
    "headers",
    [
        [("Authorization", "Bearer one"), ("Authorization", "Bearer two")],
        [("Authorization", "Bearer one"), ("X-API-Key", "two")],
    ],
)
def test_http_refuses_ambiguous_credentials(serving, headers):
    response = client_for(UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)).get(
        "/router/runtime-state/preprocessing", headers=headers
    )
    assert response.status_code == 503


def test_http_refuses_custom_config_implementation(serving):
    class CustomConfig(proxy.ProxyConfig):
        pass

    with patch.object(proxy, "proxy_config", CustomConfig()):
        response = client_for(UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)).get(
            "/router/runtime-state/preprocessing"
        )
    assert response.status_code == 503


def test_http_refuses_coerced_dispatch_flags_and_unknown_registry_types(serving):
    client = client_for(UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN))
    with patch.object(serving.router_general_settings, "pass_through_all_models", "true"):
        assert client.get("/router/runtime-state/preprocessing").status_code == 503
    with patch.object(litellm, "input_callback", None):
        assert client.get("/router/runtime-state/preprocessing").status_code == 503
