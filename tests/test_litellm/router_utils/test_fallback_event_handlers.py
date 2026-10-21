import json

import pytest

from litellm.router_utils.fallback_event_handlers import (
    AttemptedFallbackTargets,
    fallback_attempt_key,
    fallback_lookup_groups,
    get_fallback_model_group,
    get_fallback_model_group_for_lookup_groups,
    request_fallback_targets,
    run_async_fallback,
    streaming_fallback_kwargs,
)


class StreamingWrapper:
    def __init__(self):
        self._hidden_params = {"additional_headers": {}}


class FakeRouter:
    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        return StreamingWrapper()


class AlwaysFailRouter:
    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        raise RuntimeError("fallback model also failed")


@pytest.mark.asyncio
async def test_run_async_fallback_adds_errors_when_opted_in():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        include_fallback_errors=True,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["x-litellm-attempted-fallbacks"] == 1
    assert json.loads(additional_headers["x-litellm-fallback-errors"]) == [
        {
            "message": "upstream limited request",
            "type": "RuntimeError",
            "param": None,
            "code": None,
        }
    ]


@pytest.mark.asyncio
async def test_run_async_fallback_omits_errors_without_opt_in():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["x-litellm-attempted-fallbacks"] == 1
    assert "x-litellm-fallback-errors" not in additional_headers


@pytest.mark.asyncio
async def test_run_async_fallback_raises_when_all_fallbacks_fail():
    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=AlwaysFailRouter(),
            fallback_model_group=["fallback-model"],
            original_model_group="primary-model",
            original_exception=RuntimeError("original request failed"),
            max_fallbacks=3,
            fallback_depth=0,
            include_fallback_errors=True,
        )


class RecordingRouter:
    def __init__(self):
        self.received_kwargs = None

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        self.received_kwargs = kwargs
        return StreamingWrapper()


@pytest.mark.asyncio
async def test_run_async_fallback_forwards_include_fallback_errors_to_nested_call():
    """A nested fallback (multi-hop) must keep collecting errors, so the opt-in
    flag has to reach the nested async_function_with_fallbacks call."""
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        include_fallback_errors=True,
    )

    assert router.received_kwargs.get("include_fallback_errors") is True


@pytest.mark.asyncio
async def test_run_async_fallback_does_not_forward_flag_without_opt_in():
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    assert "include_fallback_errors" not in router.received_kwargs


@pytest.mark.asyncio
async def test_run_async_fallback_skips_original_model_group():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["primary-model", "fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    assert response._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"] == 1


def test_get_fallback_model_group_does_not_mutate_fallbacks():
    """A string fallback must be resolved without mutating the caller's
    fallbacks list, which is the live router config shared across requests."""
    fallbacks = [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="unmatched-model")

    assert fallback_model_group == ["gpt-4o-mini"]
    assert fallbacks == [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]


class RecordingFailRouter:
    def __init__(self):
        self.attempted_models = []

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        self.attempted_models.append(kwargs.get("model"))
        raise RuntimeError("fallback model also failed")


@pytest.mark.asyncio
async def test_run_async_fallback_skips_model_group_already_attempted():
    router = RecordingFailRouter()
    with pytest.raises(RuntimeError, match="original failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["already-attempted"],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=3,
            fallback_depth=0,
            attempted_targets=AttemptedFallbackTargets(frozenset({"already-attempted"})),
        )
    assert router.attempted_models == []


@pytest.mark.asyncio
async def test_run_async_fallback_attempts_a_repeated_target_once():
    router = RecordingFailRouter()
    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["fallback-model", "fallback-model", "other-model"],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=5,
            fallback_depth=0,
        )
    assert router.attempted_models == ["fallback-model", "other-model"]


@pytest.mark.asyncio
async def test_run_async_fallback_forwards_attempted_model_groups_to_nested_call():
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
        attempted_targets=AttemptedFallbackTargets(frozenset({"earlier-model"})),
    )
    assert router.received_kwargs["attempted_targets"].keys == frozenset(
        {"earlier-model", "primary-model", "fallback-model"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry",
    [{"model": "primary-model", "_target_order": 2}, {"model": "primary-model", "_excluded_deployment_ids": ["dep-1"]}],
)
async def test_run_async_fallback_still_retargets_the_same_group_via_dict_entry(entry):
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[entry],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
        attempted_targets=AttemptedFallbackTargets(frozenset({"primary-model"})),
    )
    assert router.received_kwargs["model"] == "primary-model"


@pytest.mark.asyncio
async def test_run_async_fallback_skips_a_repeated_dict_target():
    router = RecordingFailRouter()
    with pytest.raises(RuntimeError, match="original failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=[{"model": "already-attempted"}],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=3,
            fallback_depth=0,
            attempted_targets=AttemptedFallbackTargets(frozenset({"already-attempted"})),
        )
    assert router.attempted_models == []


@pytest.mark.asyncio
async def test_run_async_fallback_attempts_a_repeated_dict_target_once():
    router = RecordingFailRouter()
    entry = {"model": "fallback-model", "messages": [{"role": "user", "content": "shorter"}]}
    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=[entry, entry, {"model": "other-model"}],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=5,
            fallback_depth=0,
        )
    assert router.attempted_models == ["fallback-model", "other-model"]


@pytest.mark.asyncio
async def test_run_async_fallback_keeps_a_request_override_distinct_from_the_bare_name():
    router = RecordingFailRouter()
    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=[{"model": "already-attempted", "messages": [{"role": "user", "content": "shorter"}]}],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=3,
            fallback_depth=0,
            attempted_targets=AttemptedFallbackTargets(frozenset({"already-attempted"})),
        )
    assert router.attempted_models == ["already-attempted"]


@pytest.mark.parametrize(
    "target, expected", [("group-a", "group-a"), ({"model": "group-a"}, "group-a"), (None, None), (["group-a"], None)]
)
def test_fallback_attempt_key_identity(target, expected):
    assert fallback_attempt_key(target) == expected


def test_fallback_attempt_key_gives_a_param_only_entry_its_own_identity():
    key = fallback_attempt_key({"messages": [{"role": "user", "content": "shorter"}]})
    assert key is not None
    assert key != fallback_attempt_key({"messages": [{"role": "user", "content": "other"}]})


def test_fallback_attempt_key_separates_overrides_from_the_bare_name():
    bare = fallback_attempt_key("group-a")
    override = fallback_attempt_key({"model": "group-a", "messages": [{"role": "user", "content": "x"}]})
    other_override = fallback_attempt_key({"model": "group-a", "messages": [{"role": "user", "content": "y"}]})
    order_retarget = fallback_attempt_key({"model": "group-a", "_target_order": 2})
    assert len({bare, override, other_override, order_retarget}) == 4


def test_fallback_attempt_key_is_stable_across_key_order():
    assert fallback_attempt_key({"model": "group-a", "_target_order": 2}) == fallback_attempt_key(
        {"_target_order": 2, "model": "group-a"}
    )


@pytest.mark.parametrize(
    "extra, expected",
    [([], ["fb1", "fb2"]), ([{"fb1": []}], ["fb1", "fb2"]), ([{"fb1": ["own-backup"]}], ["own-backup"])],
)
@pytest.mark.parametrize("generic_first", [False, True])
def test_streaming_hop_resumes_original_chain_after_specific_lookup(extra, expected, generic_first):
    kwargs = {"attempted_targets": request_fallback_targets({}, "primary")}
    generic = [{"*": ["generic-backup"]}]
    specific = [{"primary": ["fb1", "fb2"]}, *extra]
    fallbacks = generic + specific if generic_first else specific + generic
    assert fallback_lookup_groups(kwargs, "fb1") == ("fb1", "primary")
    resolved, _ = get_fallback_model_group_for_lookup_groups(fallbacks, fallback_lookup_groups(kwargs, "fb1"))
    assert resolved == expected


def test_streaming_controls_share_tracker_without_trusting_a_caller_dict():
    controls = {"fallbacks": [{"primary": ["backup"]}], "num_retries": 0}
    tracker = request_fallback_targets(
        {**controls, "attempted_targets": {"original_model_group": "injected"}}, "primary"
    )
    tracker.record("backup")
    snapshot = streaming_fallback_kwargs("backup", {"attempted_targets": tracker})
    assert snapshot["fallbacks"] is controls["fallbacks"]
    assert snapshot["num_retries"] == 0
    assert snapshot["attempted_targets"] is tracker
    assert request_fallback_targets(snapshot, "backup") is tracker
    assert fallback_lookup_groups(snapshot, "backup") == ("backup", "primary")
    assert "backup" in tracker


def test_direct_request_with_empty_chain_does_not_inherit_another_groups_fallbacks():
    kwargs = {"attempted_targets": request_fallback_targets({}, "fb1")}
    fallbacks = [{"primary": ["fb1", "fb2"]}, {"fb1": []}, {"*": ["generic-backup"]}]
    resolved, _ = get_fallback_model_group_for_lookup_groups(fallbacks, fallback_lookup_groups(kwargs, "fb1"))
    assert resolved == []


def test_empty_candidate_chain_resumes_the_original_generic_chain():
    kwargs = {"attempted_targets": request_fallback_targets({}, "primary")}
    fallbacks = [{"*": ["fb1", "fb2"]}, {"fb1": []}]
    resolved, _ = get_fallback_model_group_for_lookup_groups(fallbacks, fallback_lookup_groups(kwargs, "fb1"))
    assert resolved == ["fb1", "fb2"]


def test_empty_original_chain_still_blocks_a_candidates_generic_chain():
    kwargs = {"attempted_targets": request_fallback_targets({}, "primary")}
    fallbacks = [{"*": ["generic-backup"]}, {"primary": []}]
    resolved, _ = get_fallback_model_group_for_lookup_groups(fallbacks, fallback_lookup_groups(kwargs, "fb1"))
    assert resolved == []
