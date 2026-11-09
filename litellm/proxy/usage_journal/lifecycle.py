from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Coroutine, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from types import MappingProxyType
from typing import ParamSpec, TypeVar
from uuid import uuid4

import httpx
from fastapi import HTTPException, Request
from prisma.errors import PrismaError
from pydantic import BaseModel, TypeAdapter, ValidationError
from starlette.responses import StreamingResponse

from .protocol import Attempt, KnownUsage, UnknownUsage
from .runtime_types import IncomingHeaders
from .store import Admission, AttemptRow, RequestRow, Store

logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")
OBJECTS = TypeAdapter(Mapping[str, object])


def fields(value: object) -> Mapping[str, object]:
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_unset=True)
    if isinstance(value, Mapping):
        return OBJECTS.validate_python(value)
    return MappingProxyType({})


def usage_from(value: object) -> KnownUsage | UnknownUsage:
    usage = fields(fields(value).get("usage"))
    if "prompt_tokens" in usage and "completion_tokens" in usage:
        prompt, completion = (usage["prompt_tokens"], usage["completion_tokens"])
        cached = fields(usage.get("prompt_tokens_details")).get("cached_tokens", 0)
    elif "input_tokens" in usage and "output_tokens" in usage:
        cached = usage.get("cache_read_input_tokens", fields(usage.get("input_tokens_details")).get("cached_tokens", 0))
        input_count = usage["input_tokens"]
        creation = usage.get("cache_creation_input_tokens", 0)
        prompt = (
            input_count + creation + cached
            if isinstance(input_count, int)
            and isinstance(creation, int)
            and isinstance(cached, int)
            and ("cache_read_input_tokens" in usage)
            else input_count
        )
        completion = usage["output_tokens"]
    else:
        return UnknownUsage(state="unknown", reason="provider_usage_missing")
    try:
        return KnownUsage.model_validate(
            MappingProxyType(
                {"state": "known", "promptTokens": prompt, "completionTokens": completion, "cacheReadTokens": cached}
            )
        )
    except ValidationError:
        return UnknownUsage(state="unknown", reason="provider_usage_invalid")


class StreamEvidence(BaseModel):
    completed_response: BaseModel | Mapping[str, object] | None = None
    chunks: Sequence[object] = ()


def stream_usage(result: object) -> KnownUsage | UnknownUsage:
    try:
        evidence = StreamEvidence.model_validate(result, from_attributes=True)
    except ValidationError:
        return UnknownUsage(state="unknown", reason="stream_usage_missing")
    if evidence.completed_response is not None:
        return usage_from(evidence.completed_response)
    reported = tuple(usage for chunk in evidence.chunks if isinstance((usage := usage_from(chunk)), KnownUsage))
    return reported[-1] if reported else UnknownUsage(state="unknown", reason="stream_usage_missing")


@dataclass
class AttemptHandle:
    row: AttemptRow
    result: object = None
    usage: KnownUsage | UnknownUsage = field(default_factory=lambda: UnknownUsage(state="unknown", reason="unfinished"))
    outcome: str = "failure"
    streaming: bool = False
    persisted: bool = False
    called: bool = False
    body: Attempt | None = None


@dataclass
class RequestHandle:
    store: Store
    row: RequestRow
    attempts: tuple[AttemptHandle, ...] = field(default_factory=tuple)


current_request: ContextVar[RequestHandle | None] = ContextVar("durable_usage_request", default=None)
current_attempt: ContextVar[AttemptHandle | None] = ContextVar("durable_usage_attempt", default=None)
stream_attempt: ContextVar[AttemptHandle | None] = ContextVar("durable_usage_stream_attempt", default=None)


@dataclass
class LifecycleState:
    store: Store | None = None
    pending: tuple[asyncio.Task[None], ...] = ()


_state = LifecycleState()
JOURNAL_ERRORS = (PrismaError, httpx.HTTPError, OSError, RuntimeError, ValueError)


def configure(store: Store | None) -> None:
    _state.store = store


def _finished(task: asyncio.Task[None]) -> None:
    _state.pending = tuple(item for item in _state.pending if item is not task)


def background(operation: Coroutine[object, object, None]) -> None:
    task = asyncio.create_task(operation)
    _state.pending += (task,)
    task.add_done_callback(_finished)


async def persist(handle: RequestHandle, attempt: AttemptHandle) -> None:
    if attempt.persisted:
        return
    body = attempt.body or Attempt.model_validate(
        MappingProxyType(
            {
                "type": "attempt.finished",
                "attemptId": attempt.row.id,
                "logId": attempt.row.log_id,
                "ordinal": attempt.row.ordinal,
                "kind": attempt.row.kind,
                "model": attempt.row.model,
                "modelGroup": attempt.row.model_group,
                "deploymentId": attempt.row.deployment_id,
                "provider": attempt.row.provider,
                "startedAt": attempt.row.started_at,
                "endedAt": datetime.now(UTC),
                "outcome": attempt.outcome,
                "usage": attempt.usage.model_dump(),
            }
        )
    )
    attempt.body = body
    await handle.store.finish(handle.row, attempt.row, body)
    attempt.persisted = True


async def persist_until_available(handle: RequestHandle, attempt: AttemptHandle) -> None:
    while not attempt.persisted:
        try:
            await persist(handle, attempt)
        except JOURNAL_ERRORS as error:
            logger.error("usage completion pending request=%s error=%s", handle.row.id, type(error).__name__)
            await asyncio.sleep(1)


async def begin_attempt(
    model: object, metadata: object, model_info: object
) -> tuple[AttemptHandle | None, Token[AttemptHandle | None] | None]:
    handle = current_request.get()
    if handle is None or current_attempt.get() is not None:
        return (None, None)
    info, meta = (fields(model_info), fields(metadata))
    name = model if isinstance(model, str) else "unknown"
    group = meta.get("model_group")
    deployment = info.get("id")
    provider = meta.get("custom_llm_provider")
    row = await handle.store.begin_attempt(
        handle.row,
        attempt_id=str(uuid4()),
        model=name,
        group=group if isinstance(group, str) else handle.row.customer_model,
        deployment=deployment if isinstance(deployment, str) else None,
        provider=provider if isinstance(provider, str) else None,
    )
    attempt = AttemptHandle(row)
    handle.attempts += (attempt,)
    return (attempt, current_attempt.set(attempt))


async def end_attempt(attempt: AttemptHandle | None, token: Token[AttemptHandle | None] | None, result: object) -> None:
    if token is not None:
        current_attempt.reset(token)
    handle = current_request.get()
    if handle is None or attempt is None:
        return
    attempt.result = result
    attempt.streaming = hasattr(result, "__aiter__")
    if attempt.streaming:
        return
    attempt.usage = (
        KnownUsage(state="known", promptTokens=0, completionTokens=0, cacheReadTokens=0)
        if result is None and (not attempt.called)
        else usage_from(result)
    )
    attempt.outcome = "success" if result is not None else "failure"
    try:
        await persist(handle, attempt)
    except JOURNAL_ERRORS as error:
        logger.error("usage finish deferred request=%s error=%s", handle.row.id, type(error).__name__)
        background(persist_until_available(handle, attempt))


async def close_request(
    handle: RequestHandle, outcome: str, wire_usage: KnownUsage | UnknownUsage | None = None
) -> None:
    for attempt in handle.attempts:
        if attempt.streaming:
            attempt.usage = (
                stream_usage(attempt.result)
                if outcome == "success" and attempt is handle.attempts[-1]
                else UnknownUsage(state="unknown", reason="stream_interrupted")
            )
            if isinstance(attempt.usage, UnknownUsage) and wire_usage is not None and (attempt is handle.attempts[-1]):
                attempt.usage = wire_usage
            attempt.outcome = outcome
            try:
                await persist(handle, attempt)
            except JOURNAL_ERRORS as error:
                logger.error("stream usage finish deferred request=%s error=%s", handle.row.id, type(error).__name__)
                background(persist_until_available(handle, attempt))
    try:
        await handle.store.close(handle.row, outcome)
    except JOURNAL_ERRORS as error:
        logger.error("request close pending request=%s error=%s", handle.row.id, type(error).__name__)
        background(close_until_available(handle, outcome))


async def close_until_available(handle: RequestHandle, outcome: str) -> None:
    while True:
        try:
            await handle.store.close(handle.row, outcome)
            return
        except JOURNAL_ERRORS:
            await asyncio.sleep(1)


@dataclass
class WireUsage:
    buffer: str = ""
    usage: Mapping[str, object] = field(default_factory=lambda: MappingProxyType(MappingProxyType({})))
    complete: bool = False
    anthropic: bool = False
    failed: bool = False

    def feed(self, chunk: str | bytes) -> None:
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
        self.buffer += text
        if len(self.buffer) > 1048576:
            self.buffer = ""
            return
        while "\n\n" in self.buffer:
            frame, self.buffer = self.buffer.split("\n\n", 1)
            data = "\n".join(
                line.removeprefix("data:").lstrip() for line in frame.splitlines() if line.startswith("data:")
            )
            if data == "[DONE]":
                self.complete = True
                continue
            try:
                payload = OBJECTS.validate_json(data)
            except (ValueError, ValidationError):
                continue
            kind = payload.get("type")
            if kind == "error" or payload.get("error"):
                self.failed = True
            if kind == "message_start":
                self.anthropic = True
                self.usage = MappingProxyType(
                    MappingProxyType({**self.usage, **fields(fields(payload.get("message")).get("usage"))})
                )
            elif kind == "message_delta":
                self.usage = MappingProxyType(MappingProxyType({**self.usage, **fields(payload.get("usage"))}))
            elif kind == "message_stop":
                self.complete = True
            elif kind == "response.completed":
                self.usage = MappingProxyType(
                    MappingProxyType({**self.usage, **fields(fields(payload.get("response")).get("usage"))})
                )
                self.complete = True
            elif payload.get("usage"):
                self.usage = MappingProxyType(MappingProxyType({**self.usage, **fields(payload.get("usage"))}))

    def result(self) -> KnownUsage | UnknownUsage:
        if self.anthropic and (not self.complete):
            return UnknownUsage(state="unknown", reason="anthropic_stream_interrupted")
        return usage_from(MappingProxyType({"usage": self.usage}))


async def body_stream(
    iterator: AsyncIterable[str | bytes | memoryview], handle: RequestHandle
) -> AsyncIterator[str | bytes]:
    outcome = "cancelled"
    request_token = current_request.set(handle)
    attempt_token = stream_attempt.set(handle.attempts[-1] if handle.attempts else None)
    wire = WireUsage()
    try:
        async for chunk in iterator:
            value = chunk.tobytes() if isinstance(chunk, memoryview) else chunk
            wire.feed(value)
            yield value
        outcome = "failure" if wire.failed else "success"
    finally:
        try:
            await asyncio.shield(close_request(handle, outcome, wire.result()))
        finally:
            current_request.reset(request_token)
            stream_attempt.reset(attempt_token)


def durable_request(function: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:

    @wraps(function)
    async def wrapper(
        *args: P.args,
        **kwargs: P.kwargs,  # kwargs-ok: ParamSpec preserves the wrapped method contract
    ) -> R:
        request = kwargs.get("request")
        if not isinstance(request, Request) or not request.headers.get("x-guangrouter-usage-protocol"):
            return await function(*args, **kwargs)
        from .runtime import authorize

        incoming = IncomingHeaders(TypeAdapter(Mapping[str, str]).validate_python(request.headers))
        authorize(incoming)
        if _state.store is None:
            raise HTTPException(503, "Durable usage journal is unavailable")
        admission = Admission(
            request_id=request.headers.get("x-guangrouter-usage-request", ""),
            trace_id=request.headers.get("x-idempotency-key", ""),
            project_id=request.headers.get("x-project-id", ""),
            customer_model=request.headers.get("x-guangrouter-customer-model", ""),
            mode=request.headers.get("x-guangrouter-usage-mode", ""),
        )
        try:
            row = await _state.store.start(admission)
        except JOURNAL_ERRORS as error:
            logger.error("usage admission refused error=%s", type(error).__name__)
            raise HTTPException(503, "Durable usage admission refused") from None
        handle = RequestHandle(_state.store, row)
        token = current_request.set(handle)
        try:
            response = await function(*args, **kwargs)
            if isinstance(response, StreamingResponse):
                response.body_iterator = body_stream(response.body_iterator, handle)
            else:
                await close_request(handle, "success")
            return response
        except BaseException:
            await asyncio.shield(close_request(handle, "failure"))
            raise
        finally:
            current_request.reset(token)

    return wrapper
