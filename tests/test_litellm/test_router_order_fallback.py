"""
Tests for order-based fallback routing.

When deployments have `order` set in litellm_params, lower order deployments
should be tried first, and higher order deployments should be used as fallbacks
when lower order deployments fail.
"""

import json
from typing import Final, Optional

import httpx
import pytest
from openai import AsyncOpenAI

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.prompt_caching_cache import PromptCachingCache
from litellm.types.router import RouterRateLimitError
from litellm.utils import _get_deployment_order, _get_order_filtered_deployments

# ---------------------------------------------------------------------------
# Unit tests for _get_order_filtered_deployments
# ---------------------------------------------------------------------------


class TestGetOrderFilteredDeployments:
    def _make_deployment(self, order: Optional[int], dep_id: str) -> dict:
        params: dict = {"model": "gpt-4o", "api_key": "key"}
        if order is not None:
            params["order"] = order
        return {
            "model_name": "test-model",
            "litellm_params": params,
            "model_info": {"id": dep_id},
        }

    def test_returns_min_order_group(self):
        deps = [
            self._make_deployment(1, "a"),
            self._make_deployment(2, "b"),
            self._make_deployment(1, "c"),
        ]
        result = _get_order_filtered_deployments(deps)
        assert len(result) == 2
        assert all(d["model_info"]["id"] in ("a", "c") for d in result)

    def test_target_order_filters_to_exact_level(self):
        deps = [
            self._make_deployment(1, "a"),
            self._make_deployment(2, "b"),
            self._make_deployment(3, "c"),
        ]
        result = _get_order_filtered_deployments(deps, target_order=2)
        assert len(result) == 1
        assert result[0]["model_info"]["id"] == "b"

    def test_target_order_no_match_returns_empty(self):
        deps = [
            self._make_deployment(1, "a"),
            self._make_deployment(2, "b"),
        ]
        result = _get_order_filtered_deployments(deps, target_order=99)
        assert result == []

    def test_no_order_set_returns_all(self):
        deps = [
            self._make_deployment(None, "a"),
            self._make_deployment(None, "b"),
        ]
        result = _get_order_filtered_deployments(deps)
        assert len(result) == 2

    def test_empty_list(self):
        result = _get_order_filtered_deployments([])
        assert result == []

    def test_single_order_returns_all_with_that_order(self):
        deps = [
            self._make_deployment(1, "a"),
            self._make_deployment(1, "b"),
        ]
        result = _get_order_filtered_deployments(deps)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration tests for order-based fallback in Router
# ---------------------------------------------------------------------------


def test_router_order_without_pre_call_checks():
    """Order filtering should work even when enable_pre_call_checks=False (default)."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "from order 1",
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "from order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
        enable_pre_call_checks=False,
    )

    for _ in range(20):
        response = router.completion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response._hidden_params["model_id"] == "1"


def test_router_order_no_fallback_when_healthy():
    """When order=1 is healthy, order=2 should never be used."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "from order 1",
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "from order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
    )

    for _ in range(50):
        response = router.completion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response._hidden_params["model_id"] == "1"


@pytest.mark.asyncio
async def test_router_order_fallback_on_failure():
    """When order=1 fails, order=2 should be tried as fallback."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad-key",
                    "mock_response": Exception("connection error"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good-key",
                    "mock_response": "success from order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "2"


@pytest.mark.asyncio
async def test_router_order_fallback_three_levels():
    """When order=1 and order=2 both fail, order=3 should be tried."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail 2"),
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "success from order 3",
                    "order": 3,
                },
                "model_info": {"id": "3"},
            },
        ],
        num_retries=0,
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "3"


@pytest.mark.asyncio
async def test_router_order_fallback_then_external_fallback():
    """When all order levels fail, external fallbacks should be tried."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 2"),
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "success from external fallback",
                },
                "model_info": {"id": "fallback"},
            },
        ],
        fallbacks=[{"test-model": ["fallback-model"]}],
        num_retries=0,
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "fallback"


@pytest.mark.asyncio
async def test_router_order_fallback_with_non_standard_fallbacks():
    """Non-standard fallback formats (e.g. fallbacks=["model-name"]) passed
    per-request should still be tried after all order levels are exhausted."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 2"),
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "success from non-standard fallback",
                },
                "model_info": {"id": "fallback"},
            },
        ],
        num_retries=0,
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        fallbacks=["fallback-model"],  # non-standard format, passed per-request
    )
    assert response._hidden_params["model_id"] == "fallback"


@pytest.mark.asyncio
async def test_router_order_fallback_with_wildcard_model_group():
    """Wildcard model groups should also advance across order levels."""
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "good",
                    "mock_response": "success from wildcard order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
    )

    response = await router.acompletion(
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "2"


def test_check_non_standard_fallback_format():
    from litellm.router_utils.fallback_event_handlers import (
        _check_non_standard_fallback_format,
    )

    # Standard formats
    assert (
        _check_non_standard_fallback_format([{"gpt-3.5-turbo": ["claude-3-haiku"]}])
        == False
    )
    assert _check_non_standard_fallback_format([{"model": ["qwen-backup"]}]) == False
    assert (
        _check_non_standard_fallback_format(
            [{"model": ["qwen-backup"], "region": ["us-east-1"]}]
        )
        == False
    )

    # Non-standard formats
    assert _check_non_standard_fallback_format([{"model": "qwen-backup"}]) == True
    assert (
        _check_non_standard_fallback_format(
            [{"model": "qwen-backup", "messages": [{"role": "user", "content": "hi"}]}]
        )
        == True
    )
    assert (
        _check_non_standard_fallback_format(
            [{"model": ["qwen-backup"], "api_key": "some-key"}]
        )
        == True
    )


@pytest.mark.asyncio
async def test_router_order_fallback_does_not_reselect_order_1_when_order_2_is_filtered_out():

    class _DropOrder2(CustomLogger):
        async def async_filter_deployments(
            self, model, healthy_deployments, messages, request_kwargs=None, parent_otel_span=None
        ):
            return [d for d in healthy_deployments if _get_deployment_order(d) != 2]

    drop_order_2: Final = _DropOrder2()
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "litellm.RateLimitError",
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "success from order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
    )
    litellm.callbacks.append(drop_order_2)
    try:
        with pytest.raises(RouterRateLimitError, match="No deployments available") as exc_info:
            await router.acompletion(model="test-model", messages=[{"role": "user", "content": "hi"}])
        assert "success from order 2" not in str(exc_info.value)
    finally:
        litellm.callbacks.remove(drop_order_2)


@pytest.mark.asyncio
async def test_router_order_fallback_ignores_prompt_cache_pin_on_target_order():
    messages = [{"role": "user", "content": "word " * 5000}]
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("azure peak load"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "success from order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
        optional_pre_call_checks=["prompt_caching"],
    )
    await PromptCachingCache(cache=router.cache).async_add_model_id(model_id="1", messages=messages, tools=None)
    response = await router.acompletion(model="test-model", messages=messages)
    assert response._hidden_params["model_id"] == "2"


@pytest.mark.asyncio
async def test_router_order_fallback_retries_keep_target_order():
    seen_target_orders: Final = []

    class _RecordTargetOrder(CustomLogger):
        async def async_filter_deployments(
            self, model, healthy_deployments, messages, request_kwargs=None, parent_otel_span=None
        ):
            seen_target_orders.append((request_kwargs or {}).get("_target_order"))
            return healthy_deployments

    recorder: Final = _RecordTargetOrder()
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 2"),
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=1,
    )
    litellm.callbacks.append(recorder)
    try:
        with pytest.raises(Exception, match="fail order 2"):
            await router.acompletion(model="test-model", messages=[{"role": "user", "content": "hi"}])
    finally:
        litellm.callbacks.remove(recorder)
    assert seen_target_orders.count(2) >= 2


@pytest.mark.asyncio
async def test_generic_api_call_strips_target_order_from_provider_kwargs():
    captured: Final = {}

    async def _fake_provider(**provider_kwargs):
        captured.update(provider_kwargs)
        return "ok"

    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key", "order": 2},
                "model_info": {"id": "2"},
            }
        ]
    )
    response = await router._ageneric_api_call_with_fallbacks_helper(
        model="test-model",
        original_generic_function=_fake_provider,
        _target_order=2,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response == "ok"
    assert captured["model"] == "gpt-4o"
    assert "_target_order" not in captured


@pytest.mark.asyncio
async def test_text_completion_order_fallback_hop_does_not_send_target_order_upstream():
    upstream_bodies: Final[list[dict]] = []

    def _upstream(request: httpx.Request) -> httpx.Response:
        upstream_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 0,
                "model": "gpt-3.5-turbo-instruct",
                "choices": [{"text": "ok from order 2", "index": 0, "logprobs": None, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    upstream_client: Final = AsyncOpenAI(
        api_key="key",
        base_url="http://upstream.test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_upstream)),
    )
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "text-completion-openai/gpt-3.5-turbo-instruct",
                    "api_key": "key",
                    "mock_response": Exception("fail order 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "text-completion-openai/gpt-3.5-turbo-instruct",
                    "api_key": "key",
                    "api_base": "http://upstream.test",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
    )
    try:
        response = await router.atext_completion(model="test-model", prompt="hi", client=upstream_client)
    finally:
        await upstream_client.close()
    assert response._hidden_params["model_id"] == "2"
    assert upstream_bodies
    assert all(("_target_order" not in body for body in upstream_bodies))


@pytest.mark.asyncio
@pytest.mark.parametrize("external_fallbacks", [False, True])
@pytest.mark.parametrize("blocked_order", [False, True])
async def test_order_fallback_attempts_each_target_once_over_http(
    external_fallbacks: bool, blocked_order: bool
) -> None:
    attempts: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        attempts.append(body["model"])
        assert "attempted_targets" not in body
        assert "_target_order" not in body
        return httpx.Response(500, json={"error": {"message": "deployment failed", "type": "server_error"}})

    groups = (("openai/group-a", (1, 2, 3)), ("openai/group-b", (1, 2, 3)), ("openai/group-c", (1,)))
    deployments = [
        {
            "model_name": group,
            "litellm_params": {"model": f"openai/{group.split('/')[-1]}-{order}", "api_key": "key", "order": order},
            "model_info": {"id": f"{group}-{order}"},
        }
        for group, orders in groups
        for order in orders
    ] + (
        [
            {
                "model_name": "openai/group-a",
                "litellm_params": {"model": "openai/blocked", "api_key": "key", "order": 4},
                "model_info": {"id": "blocked", "blocked": True},
            }
        ]
        if blocked_order
        else []
    )
    router = Router(
        model_list=deployments,
        model_group_alias={"group-a": "openai/group-a"},
        fallbacks=[
            {"group-a": ["openai/group-b", "openai/group-c"] if external_fallbacks else []},
            {"openai/group-a": []},
            {"openai/group-b": []},
            {"openai/group-c": []},
        ],
        num_retries=0,
        max_fallbacks=10,
        disable_cooldowns=True,
    )
    async with AsyncOpenAI(
        api_key="key",
        base_url="http://upstream.test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
    ) as client:
        try:
            for _ in range(2):
                with pytest.raises((litellm.InternalServerError, RouterRateLimitError)):
                    await router.acompletion(
                        model="group-a", messages=[{"role": "user", "content": "hi"}], client=client
                    )
        finally:
            router.reset()
    assert (
        attempts
        == (
            ["group-a-1", "group-a-2", "group-a-3"]
            + (["group-b-1", "group-b-2", "group-b-3", "group-c-1"] if external_fallbacks else [])
        )
        * 2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("request_overrides", [False, True])
@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("primary_streams", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
async def test_streaming_fallback_walks_remaining_targets_over_http(
    endpoint: str,
    request_overrides: bool,
    ordered: bool,
    primary_streams: bool,
    exhausted: bool,
    respx_mock,
    monkeypatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    attempts: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        attempts.append(model)
        assert set(body) <= {"model", "messages", "input", "stream", "stream_options", "metadata"}
        assert "attempted_targets" not in body.get("metadata", {})
        if endpoint == "chat":
            assert body["messages"] == [{"role": "user", "content": "hi"}]
        else:
            assert body["input"] == "hi"
        if model.startswith("group-a-") and not primary_streams:
            return httpx.Response(500, json={"error": {"message": "primary unavailable", "type": "server_error"}})
        if not model.startswith("group-d-") or exhausted:
            error = {"message": "stream overloaded", "type": "server_error", "code": "500"}
            event = {"error": error} if endpoint == "chat" else {"type": "error", "error": error}
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, text=f"data: {json.dumps(event)}\n\n"
            )
        if endpoint == "chat":
            event = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
            }
        else:
            event = {
                "type": "response.completed",
                "sequence_number": 0,
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 0,
                    "model": model,
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
        )

    groups = (("a", (1, 2, 3) if ordered else (1,)), ("b", (1, 2, 3) if ordered else (1,)), ("c", (1,)), ("d", (1,)))
    deployments = [
        {
            "model_name": f"openai/group-{group}",
            "litellm_params": {
                "model": f"openai/group-{group}-{order}",
                "api_key": "key",
                "api_base": "http://upstream.test/v1",
                "order": order,
            },
            "model_info": {"id": f"{group}-{order}"},
        }
        for group, orders in groups
        for order in orders
    ]
    fallbacks = [{"group-a": ["openai/group-b", "openai/group-c", "openai/group-d"]}]
    router = Router(
        model_list=deployments,
        model_group_alias={"group-a": "openai/group-a"},
        fallbacks=[{"group-a": ["missing-group"]}] if request_overrides else fallbacks,
        num_retries=0,
        max_fallbacks=20,
        disable_cooldowns=True,
    )
    if endpoint == "responses":
        respx_mock.post("http://upstream.test/v1/responses").mock(side_effect=upstream)
    async with AsyncOpenAI(
        api_key="key",
        base_url="http://upstream.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
    ) as client:
        try:
            for _ in range(2):
                overrides = {"fallbacks": fallbacks} if request_overrides else {}

                async def consume() -> tuple:
                    if endpoint == "chat":
                        response = await router.acompletion(
                            model="group-a",
                            messages=[{"role": "user", "content": "hi"}],
                            stream=True,
                            client=client,
                            **overrides,
                        )
                    else:
                        response = await router.aresponses(
                            model="group-a",
                            input="hi",
                            stream=True,
                            client=client,
                            **overrides,
                        )
                    return tuple([item async for item in response])

                if exhausted:
                    with pytest.raises(
                        (litellm.APIError, litellm.APIConnectionError, litellm.InternalServerError),
                        match="stream overloaded",
                    ):
                        await consume()
                else:
                    chunks = await consume()
                    if endpoint == "chat":
                        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "ok"
                    else:
                        assert chunks[-1].type == "response.completed"
                        assert chunks[-1].response.model == "group-d-1"
        finally:
            router.reset()
    assert attempts == [f"group-{group}-{order}" for group, orders in groups for order in orders] * 2
