import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import litellm
from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.sensitive_data_masker import mask_sensitive_structure
from litellm.router_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    get_fallback_error_info,
)
from litellm.types.router import LiteLLMParamsTypedDict

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


def fallback_attempt_key(fallback_target: object) -> str | None:
    if isinstance(fallback_target, str):
        return fallback_target
    if not isinstance(fallback_target, dict):
        return None
    model: Final = fallback_target.get("model")
    if tuple(fallback_target) == ("model",) and isinstance(model, str):
        return model
    serialized: Final = json.dumps(fallback_target, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass(slots=True)
class AttemptedFallbackTargets:
    keys: frozenset[str] = frozenset()
    original_model_group: str | None = None
    overrides: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def record(self, key: str) -> None:
        self.keys = self.keys | frozenset((key,))


def request_fallback_targets(kwargs: Mapping[str, object], model_group: str | None) -> AttemptedFallbackTargets:
    carried: Final = kwargs.get("attempted_targets")
    if isinstance(carried, AttemptedFallbackTargets):
        return carried
    controls: Final = (
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "num_retries",
        "model_group_retry_policy",
        "disable_fallbacks",
    )
    return AttemptedFallbackTargets(
        original_model_group=model_group,
        overrides=MappingProxyType({key: kwargs[key] for key in controls if key in kwargs}),
    )


def streaming_fallback_kwargs(model: str, kwargs: Mapping[str, object]) -> dict[str, object]:
    carried: Final = kwargs.get("attempted_targets")
    overrides: Final = carried.overrides if isinstance(carried, AttemptedFallbackTargets) else kwargs
    return {**kwargs, **overrides, "model": model}


def fallback_lookup_groups(kwargs: Mapping[str, object], model_group: str | None) -> tuple[str, ...]:
    carried: Final = kwargs.get("attempted_targets")
    original_group: Final = carried.original_model_group if isinstance(carried, AttemptedFallbackTargets) else None
    return tuple(dict.fromkeys(group for group in (model_group, original_group) if group))


def get_fallback_model_group_for_lookup_groups(
    fallbacks: Sequence[str | Mapping[str, Sequence[str]]], lookup_groups: tuple[str, ...]
) -> tuple[list[str] | None, int | None]:
    results: Final = tuple(get_fallback_model_group(fallbacks, group) for group in lookup_groups)
    specific: Final = next(
        (result for result in results if result[0] and _resolved_specific_chain(fallbacks, result)), None
    )
    if specific is not None:
        return specific
    return next((result for result in reversed(results) if result[0] is not None), (None, None))


def _resolved_specific_chain(
    fallbacks: Sequence[str | Mapping[str, Sequence[str]]], result: tuple[Sequence[str] | None, int | None]
) -> bool:
    resolved, generic_idx = result
    if resolved is None:
        return False
    if generic_idx is None:
        return True
    generic_entry: Final = fallbacks[generic_idx]
    return not isinstance(generic_entry, dict) or resolved is not generic_entry.get("*")


def _check_stripped_model_group(model_group: str, fallback_key: str) -> bool:
    """
    Handles wildcard routing scenario

    where fallbacks set like:
    [{"gpt-3.5-turbo": ["claude-3-haiku"]}]

    but model_group is like:
    "openai/gpt-3.5-turbo"

    Returns:
    - True if the stripped model group == fallback_key
    """
    for provider in litellm.provider_list:
        if isinstance(provider, Enum):
            _provider = provider.value
        else:
            _provider = provider
        if model_group.startswith(f"{_provider}/"):
            stripped_model_group = model_group.replace(f"{_provider}/", "")
            if stripped_model_group == fallback_key:
                return True
    return False


def get_fallback_model_group(fallbacks: Sequence[Any], model_group: str) -> tuple[list[str] | None, int | None]:
    """
    Returns:
    - fallback_model_group: List[str] of fallback model groups. example: ["gpt-4", "gpt-3.5-turbo"]
    - generic_fallback_idx: int of the index of the generic fallback in the fallbacks list.

    Checks:
    - exact match
    - stripped model group match
    - generic fallback
    """
    generic_fallback_idx: int | None = None
    stripped_model_fallback: list[str] | None = None
    fallback_model_group: list[str] | None = None
    ## check for specific model group-specific fallbacks
    for idx, item in enumerate(fallbacks):
        if isinstance(item, dict):
            if list(item.keys())[0] == model_group:  # check exact match
                fallback_model_group = item[model_group]
                break
            elif _check_stripped_model_group(
                model_group=model_group, fallback_key=list(item.keys())[0]
            ):  # check generic fallback
                stripped_model_fallback = item[list(item.keys())[0]]
            elif list(item.keys())[0] == "*":  # check generic fallback
                generic_fallback_idx = idx
        elif isinstance(item, str):
            fallback_model_group = [item]
    ## if none, check for generic fallback
    if fallback_model_group is None:
        if stripped_model_fallback is not None:
            fallback_model_group = stripped_model_fallback
        elif generic_fallback_idx is not None:
            fallback_model_group = fallbacks[generic_fallback_idx]["*"]

    return fallback_model_group, generic_fallback_idx


async def run_async_fallback(
    *args: tuple[Any],
    litellm_router: LitellmRouter,
    fallback_model_group: list[str],
    original_model_group: str,
    original_exception: Exception,
    max_fallbacks: int,
    fallback_depth: int,
    include_fallback_errors: bool = False,
    **kwargs,
) -> Any:
    """
    Loops through all the fallback model groups and calls kwargs["original_function"] with the arguments and keyword arguments provided.

    If the call is successful, it logs the success and returns the response.
    If the call fails, it logs the failure and continues to the next fallback model group.
    If all fallback model groups fail, it raises the most recent exception.

    Args:
        litellm_router: The litellm router instance.
        *args: Positional arguments.
        fallback_model_group: List[str] of fallback model groups. example: ["gpt-4", "gpt-3.5-turbo"]
        original_model_group: The original model group. example: "gpt-3.5-turbo"
        original_exception: The original exception.
        **kwargs: Keyword arguments.

    Returns:
        The response from the successful fallback model group.
    Raises:
        The most recent exception if all fallback model groups fail.
    """

    ### BASE CASE ### MAX FALLBACK DEPTH REACHED
    if fallback_depth >= max_fallbacks:
        raise original_exception

    error_from_fallbacks = original_exception
    fallback_errors = (get_fallback_error_info(original_exception),)
    carried_targets: Final = kwargs.get("attempted_targets")
    attempted: Final = (
        carried_targets if isinstance(carried_targets, AttemptedFallbackTargets) else AttemptedFallbackTargets()
    )
    attempted.record(original_model_group)

    for mg in fallback_model_group:
        if mg == original_model_group:
            continue
        attempt_key = fallback_attempt_key(mg)
        if attempt_key is not None:
            if attempt_key in attempted:
                verbose_router_logger.info(
                    "Skipping fallback to model_group = %s, already attempted for this request",
                    mask_sensitive_structure(mg),
                )
                continue
            attempted.record(attempt_key)
        try:
            # LOGGING
            kwargs = litellm_router.log_retry(kwargs=kwargs, e=original_exception)
            verbose_router_logger.info(f"Falling back to model_group = {mask_sensitive_structure(mg)}")
            kwargs.pop("_target_order", None)
            if isinstance(mg, str):
                kwargs["model"] = mg
            elif isinstance(mg, dict):
                kwargs.update(mg)
            kwargs.setdefault("metadata", {}).update(
                {"model_group": kwargs.get("model", None)}
            )  # update model_group used, if fallbacks are done
            fallback_depth = fallback_depth + 1
            kwargs["fallback_depth"] = fallback_depth
            kwargs["max_fallbacks"] = max_fallbacks
            kwargs["attempted_targets"] = attempted
            if include_fallback_errors:
                kwargs["include_fallback_errors"] = include_fallback_errors
            response = await litellm_router.async_function_with_fallbacks(*args, **kwargs)
            verbose_router_logger.info("Successful fallback b/w models.")
            response = add_fallback_headers_to_response(
                response=response,
                attempted_fallbacks=fallback_depth,
                fallback_errors=(list(fallback_errors) if include_fallback_errors else None),
            )
            # callback for successfull_fallback_event():
            await log_success_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
            return response
        except Exception as e:
            error_from_fallbacks = e
            fallback_errors = fallback_errors + (get_fallback_error_info(e),)
            await log_failure_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
    raise error_from_fallbacks


async def log_success_fallback_event(original_model_group: str, kwargs: dict, original_exception: Exception):
    """
    Log a successful fallback event to all registered callbacks.

    Uses LoggingCallbackManager.get_custom_loggers_for_type() to get deduplicated
    CustomLogger instances from all callback lists.

    Args:
        original_model_group (str): The original model group before fallback.
        kwargs (dict): kwargs for the request

    Note:
        Errors during logging are caught and reported but do not interrupt the process.
    """
    # Get deduplicated CustomLogger instances from all callback lists
    custom_loggers = litellm.logging_callback_manager.get_custom_loggers_for_type(CustomLogger)

    for _callback_custom_logger in custom_loggers:
        try:
            await _callback_custom_logger.log_success_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
        except Exception as e:
            verbose_router_logger.error(f"Error in log_success_fallback_event: {e!s}")


async def log_failure_fallback_event(original_model_group: str, kwargs: dict, original_exception: Exception):
    """
    Log a failed fallback event to all registered callbacks.

    Uses LoggingCallbackManager.get_custom_loggers_for_type() to get deduplicated
    CustomLogger instances from all callback lists.

    Args:
        original_model_group (str): The original model group before fallback.
        kwargs (dict): kwargs for the request

    Note:
        Errors during logging are caught and reported but do not interrupt the process.
    """
    # Get deduplicated CustomLogger instances from all callback lists
    custom_loggers = litellm.logging_callback_manager.get_custom_loggers_for_type(CustomLogger)

    for _callback_custom_logger in custom_loggers:
        try:
            await _callback_custom_logger.log_failure_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
        except Exception as e:
            verbose_router_logger.error(f"Error in log_failure_fallback_event: {e!s}")


def _check_non_standard_fallback_format(fallbacks: Sequence[Any] | None) -> bool:
    """
    Checks if the fallbacks list is a list of strings or a list of dictionaries.

    If
    - List[str]: e.g. ["claude-3-haiku", "openai/o-1"]
    - List[Dict[<LiteLLMParamsTypedDict>, Any]]: e.g. [{"model": "claude-3-haiku", "messages": [{"role": "user", "content": "Hey, how's it going?"}]}]

    If [{"gpt-3.5-turbo": ["claude-3-haiku"]}] then standard format.
    """
    if fallbacks is None or not isinstance(fallbacks, list) or len(fallbacks) == 0:
        return False
    if all(isinstance(item, str) for item in fallbacks):
        return True
    elif all(isinstance(item, dict) for item in fallbacks):
        for item in fallbacks:
            for key in LiteLLMParamsTypedDict.__annotations__.keys():
                if key in item:
                    # If the value is a list, it's likely a standard fallback model group mapping
                    # (e.g. {"model": ["backup"]}) rather than a parameter override.
                    if not isinstance(item[key], list):
                        return True

    return False


def run_non_standard_fallback_format(fallbacks: list[str] | list[dict[str, Any]], model_group: str):
    pass
