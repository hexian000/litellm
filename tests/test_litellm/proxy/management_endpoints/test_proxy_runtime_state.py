import asyncio
import json
import os
import uuid
from contextlib import ExitStack
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

import litellm
from litellm.proxy import proxy_server as proxy
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import _cache_key_object
from litellm.proxy.auth.user_api_key_auth import hash_token, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.management_endpoints.proxy_runtime_state import CALLBACK_REGISTRIES
from litellm.proxy.management_endpoints.router_settings_endpoints import router
from litellm.proxy.utils import PrismaClient
from litellm.router import Router


@pytest.fixture
def serving():
    instance = Router(
        model_list=[
            {
                "model_name": "bounded",
                "model_info": {"id": "selected", "input_cost_per_token": 0.0000001, "output_cost_per_token": 0.0000001},
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


async def preprocess(auth, model="bounded", *, path="/v1/chat/completions", route_type="acompletion", body=None):
    data = body or {"model": model, "messages": [{"role": "user", "content": "local only"}], "max_tokens": 65536}
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
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
        route_type=route_type,
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


def inspection_envelope(path="/v1/chat/completions", **body):
    return {
        "path": path,
        "body": json.dumps(
            {"model": "public", "messages": [{"role": "user", "content": "local only"}], "max_tokens": 65536, **body}
        ),
    }


@pytest.mark.asyncio
async def test_admin_http_inspects_actual_cached_inference_credential_and_preserves_admin_boundary(serving):
    key = "sk-local-normal-inspection"
    admin = "sk-local-master-inspection"
    cached = UserAPIKeyAuth(
        api_key=hash_token(key),
        user_role=LitellmUserRoles.INTERNAL_USER,
        aliases={"public": "bounded"},
        router_settings={"timeout": 4},
    )
    with patch.object(proxy, "master_key", admin), patch.object(proxy, "prisma_client", object()):
        await _cache_key_object(hash_token(key), cached, proxy.user_api_key_cache, proxy.proxy_logging_obj)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app), base_url="http://127.0.0.1"
        ) as client:
            headers = {"Authorization": "Bearer " + admin, "X-LiteLLM-Inspection-Credential": "Bearer " + key}
            normal_get = await client.get(
                "/router/runtime-state/preprocessing", headers={"Authorization": "Bearer " + key}
            )
            denied = await client.post(
                "/router/runtime-state/preprocessing",
                headers={**headers, "Authorization": "Bearer " + key},
                json=inspection_envelope(),
            )
            observed = await client.post(
                "/router/runtime-state/preprocessing", headers=headers, json=inspection_envelope()
            )
            repeated = await client.post(
                "/router/runtime-state/preprocessing", headers=headers, json=inspection_envelope()
            )
            changed = await client.post(
                "/router/runtime-state/preprocessing", headers=headers, json=inspection_envelope(max_tokens=65537)
            )
            admin_get = await client.get(
                "/router/runtime-state/preprocessing", headers={"Authorization": "Bearer " + admin}
            )
            missing = await client.post(
                "/router/runtime-state/preprocessing",
                headers={"Authorization": "Bearer " + admin},
                json=inspection_envelope(),
            )
    assert normal_get.status_code == 401
    assert denied.status_code in (401, 403)
    assert observed.status_code == repeated.status_code == changed.status_code == admin_get.status_code == 200, (
        observed.text
    )
    assert missing.status_code == 400
    result = observed.json()
    assert result["protocol"] == "litellm.runtime-inference-preprocessing-state/v1"
    assert result["authentication"]["aliases"] == {"public": "bounded"}
    assert result["authentication"]["effective_router_settings"]["values"] == {"timeout": 4}
    assert admin_get.json()["authentication"]["aliases"] == {}
    assert result["authentication"]["request_fingerprint"] == repeated.json()["authentication"]["request_fingerprint"]
    assert result["authentication"]["request_fingerprint"] != changed.json()["authentication"]["request_fingerprint"]
    assert (
        result["authentication"]["credential_fingerprint"] == changed.json()["authentication"]["credential_fingerprint"]
    )
    assert observed.headers["Cache-Control"] == "no-store"
    assert (
        key not in observed.text and admin not in observed.text and "private-provider-credential" not in observed.text
    )
    assert result["authenticated_request"]["values"]["model"] == "public"
    actual = await preprocess(cached, model="public")
    assert actual["model"] == "bounded"
    assert actual["router_settings_override"] == result["authentication"]["effective_router_settings"]["values"]


@pytest.mark.parametrize(
    "envelope",
    [
        {"path": "/key/info", "body": "{}"},
        {"path": "/messages", "body": "{}"},
        {"path": "/v1/chat/completions", "body": "[]"},
        {"path": "/v1/chat/completions", "body": "{}", "headers": [["authorization", "Bearer other"]]},
        {"path": "/v1/chat/completions", "body": "{}", "headers": [["x-model", "one"], ["X-Model", "two"]]},
    ],
)
def test_inference_inspection_refuses_unsupported_or_ambiguous_request_before_authentication(serving, envelope):
    client = client_for(UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN))
    result = client.post(
        "/router/runtime-state/preprocessing",
        headers={"X-LiteLLM-Inspection-Credential": "Bearer sk-subject-only"},
        json=envelope,
    )
    assert result.status_code in (400, 422)
    assert "sk-subject-only" not in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,route_type",
    [
        ("/chat/completions", "acompletion"),
        ("/v1/chat/completions", "acompletion"),
        ("/responses", "aresponses"),
        ("/v1/responses", "aresponses"),
        ("/v1/messages", "anthropic_messages"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_actual_inference_authentication_and_preprocessing_across_supported_routes(
    serving, path, route_type, stream
):
    key = "sk-local-routes-" + uuid.uuid4().hex
    auth = UserAPIKeyAuth(
        api_key=hash_token(key),
        user_role=LitellmUserRoles.INTERNAL_USER,
        aliases={"public": "bounded"},
        router_settings={"timeout": 4},
    )
    body = {
        "model": "public",
        "stream": stream,
        **(
            {"input": "local only", "max_output_tokens": 65536}
            if route_type == "aresponses"
            else {"messages": [{"role": "user", "content": "local only"}], "max_tokens": 65536}
        ),
    }
    with patch.object(proxy, "master_key", "sk-local-route-admin"), patch.object(proxy, "prisma_client", object()):
        await _cache_key_object(hash_token(key), auth, proxy.user_api_key_cache, proxy.proxy_logging_obj)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app), base_url="http://127.0.0.1"
        ) as client:
            response = await client.post(
                "/router/runtime-state/preprocessing",
                headers={
                    "Authorization": "Bearer sk-local-route-admin",
                    "X-LiteLLM-Inspection-Credential": "Bearer " + key,
                },
                json={"path": path, "body": json.dumps(body)},
            )
        actual = await preprocess(auth, path=path, route_type=route_type, body=body)
    assert response.status_code == 200, response.text
    assert actual["model"] == response.json()["authentication"]["aliases"]["public"]
    assert actual["stream"] is stream
    assert (
        actual["router_settings_override"] == response.json()["authentication"]["effective_router_settings"]["values"]
    )


@pytest.mark.asyncio
async def test_inference_inspection_refuses_unsupported_cached_extensions_without_disclosing_them(serving):
    key = "sk-local-extension-" + uuid.uuid4().hex
    auth = UserAPIKeyAuth(
        api_key=hash_token(key), user_role=LitellmUserRoles.INTERNAL_USER, permissions={"secret-extension": True}
    )
    with patch.object(proxy, "master_key", "sk-local-route-admin"), patch.object(proxy, "prisma_client", object()):
        await _cache_key_object(hash_token(key), auth, proxy.user_api_key_cache, proxy.proxy_logging_obj)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app), base_url="http://127.0.0.1"
        ) as client:
            response = await client.post(
                "/router/runtime-state/preprocessing",
                headers={
                    "Authorization": "Bearer sk-local-route-admin",
                    "X-LiteLLM-Inspection-Credential": "Bearer " + key,
                },
                json=inspection_envelope(),
            )
    assert response.status_code == 503, response.text
    assert "secret-extension" not in response.text


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("LITELLM_INSPECTION_TEST_DATABASE_URL"), reason="requires an isolated PostgreSQL schema"
)
async def test_real_postgres_generated_team_key_inspection_and_retained_process_authentication(serving):
    database_url = os.environ["LITELLM_INSPECTION_TEST_DATABASE_URL"]
    assert "127.0.0.1" in database_url and "/inspection" in database_url
    with patch.dict(os.environ, {"DATABASE_URL": database_url}):
        db = PrismaClient(database_url=database_url, proxy_logging_obj=proxy.proxy_logging_obj)
        await db.connect()
        try:
            with (
                patch.object(proxy, "master_key", "sk-local-db-inspection-admin"),
                patch.object(proxy, "prisma_client", db),
            ):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=proxy.app),
                    base_url="http://127.0.0.1",
                    headers={"Authorization": "Bearer sk-local-db-inspection-admin"},
                ) as client:
                    team = await client.post(
                        "/team/new",
                        json={
                            "team_id": "inspection-" + uuid.uuid4().hex,
                            "models": ["bounded"],
                            "metadata": {"workbench_project_id": "local-project"},
                            "router_settings": {"timeout": 8},
                        },
                    )
                    assert team.status_code == 200, team.text
                    key = await client.post(
                        "/key/generate",
                        json={
                            "team_id": team.json()["team_id"],
                            "models": ["bounded"],
                            "metadata": {
                                "workbench_api_key": True,
                                "workbench_scene": "dev",
                                "workbench_project_id": "local-project",
                                "workbench_media_models": [],
                                "workbench_priority": "normal",
                            },
                            "max_budget": 1,
                        },
                    )
                    assert key.status_code == 200, key.text
                    bearer = key.json()["key"]
                    headers = {"X-LiteLLM-Inspection-Credential": "Bearer " + bearer}
                    envelope = inspection_envelope(model="bounded")
                    from litellm.proxy.spend_tracking.budget_reservation import release_budget_reservation

                    with patch(
                        "litellm.proxy.management_endpoints.inference_runtime_state.release_budget_reservation",
                        wraps=release_budget_reservation,
                    ) as released:
                        first = await client.post("/router/runtime-state/preprocessing", headers=headers, json=envelope)
                    reservation = released.call_args.args[0]
                    assert reservation and reservation["reserved_cost"] > 0 and reservation["finalized"] is True
                    for entry in reservation["entries"]:
                        assert await proxy.spend_counter_cache.async_get_cache(
                            key=entry["counter_key"]
                        ) == pytest.approx(0)
                    assert first.status_code == 200, first.text
                    observed = first.json()["authentication"]
                    assert observed["scopes"]["team_id"] == team.json()["team_id"]
                    assert observed["effective_router_settings"]["values"] == {"timeout": 8}
                    assert observed["metadata"]["metadata"]["keys"] == [
                        "workbench_api_key",
                        "workbench_media_models",
                        "workbench_priority",
                        "workbench_project_id",
                        "workbench_scene",
                    ]
                    assert "local-project" not in first.text
                    from litellm.proxy.management_endpoints import inference_runtime_state as inference_protocol

                    authenticated = asyncio.Event()
                    allow_return = asyncio.Event()
                    captured_auth = []
                    original_authentication = user_api_key_auth

                    async def gated_authentication(**kwargs):
                        resolved = await original_authentication(**kwargs)
                        captured_auth.append(resolved)
                        authenticated.set()
                        await allow_return.wait()
                        return resolved

                    with patch.object(inference_protocol, "user_api_key_auth", gated_authentication):
                        cancelled = asyncio.create_task(
                            client.post("/router/runtime-state/preprocessing", headers=headers, json=envelope)
                        )
                        await asyncio.wait_for(authenticated.wait(), timeout=10)
                        cancelled.cancel()
                        allow_return.set()
                        with pytest.raises(asyncio.CancelledError):
                            await cancelled
                    cancelled_reservation = captured_auth[0].budget_reservation
                    assert cancelled_reservation and cancelled_reservation["finalized"] is True
                    for entry in cancelled_reservation["entries"]:
                        assert await proxy.spend_counter_cache.async_get_cache(
                            key=entry["counter_key"]
                        ) == pytest.approx(0)
                    cached = await proxy.user_api_key_cache.async_get_cache(
                        key=hash_token(bearer), model_type=UserAPIKeyAuth
                    )
                    assert cached is not None
                    retained = cached.model_copy(
                        update={"aliases": {"public": "bounded"}, "router_settings": {"timeout": 4}}
                    )
                    await _cache_key_object(
                        hash_token(bearer), retained, proxy.user_api_key_cache, proxy.proxy_logging_obj
                    )
                    second = await client.post("/router/runtime-state/preprocessing", headers=headers, json=envelope)
                    assert second.status_code == 200, second.text
                    assert second.json()["authentication"]["aliases"] == {"public": "bounded"}
                    assert second.json()["authentication"]["effective_router_settings"]["values"] == {"timeout": 4}
                    info = await client.get("/key/info", params={"key": bearer})
                    assert info.status_code == 200, info.text
                    assert info.json()["info"]["aliases"] == {}
                    assert bearer not in first.text and bearer not in second.text
                    native_project_id = "inspection-" + uuid.uuid4().hex
                    await db.db.litellm_projecttable.create(
                        data={
                            "project_id": native_project_id,
                            "created_by": "local-inspection-fixture",
                            "updated_by": "local-inspection-fixture",
                            "team_id": team.json()["team_id"],
                            "metadata": json.dumps({"workbench_project_id": "native-project"}),
                            "models": ["bounded"],
                        }
                    )
                    native_key = await client.post(
                        "/key/generate",
                        json={
                            "project_id": native_project_id,
                            "created_by": "local-inspection-fixture",
                            "updated_by": "local-inspection-fixture",
                            "team_id": team.json()["team_id"],
                            "models": ["bounded"],
                            "metadata": {"workbench_api_key": True},
                        },
                    )
                    assert native_key.status_code == 200, native_key.text
                    native = await client.post(
                        "/router/runtime-state/preprocessing",
                        headers={"X-LiteLLM-Inspection-Credential": "Bearer " + native_key.json()["key"]},
                        json=envelope,
                    )
                    assert native.status_code == 200, native.text
                    assert native.json()["authentication"]["scopes"]["project_id"] == native_project_id
                    assert native.json()["authentication"]["metadata"]["project_metadata"]["keys"] == [
                        "workbench_project_id"
                    ]
                    blocked = await client.post("/key/block", json={"key": bearer})
                    assert blocked.status_code == 200, blocked.text
                    refused = await client.post("/router/runtime-state/preprocessing", headers=headers, json=envelope)
                    assert refused.status_code == 403, refused.text
        finally:
            await db.disconnect()
