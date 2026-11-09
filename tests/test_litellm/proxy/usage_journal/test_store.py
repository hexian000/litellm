from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from prisma import Prisma
from pydantic import TypeAdapter

from litellm.proxy.usage_journal.lifecycle import WireUsage, usage_from
from litellm.proxy.usage_journal.protocol import Attempt, KnownUsage, UnknownUsage, UsageBatch
from litellm.proxy.usage_journal.runtime import deliver
from litellm.proxy.usage_journal.store import Admission, Store


class Database:
    def __init__(self, client: Prisma) -> None:
        self.writer_db = client

    def tx(self):
        return self.writer_db.tx()


@pytest_asyncio.fixture(loop_scope="function")
async def store() -> AsyncIterator[Store]:
    url = os.environ.get("LITELLM_USAGE_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires migrated isolated PostgreSQL LITELLM_USAGE_TEST_DATABASE_URL")
    db = Prisma(datasource={"url": url})
    await db.connect()
    try:
        yield Store(Database(db), f"test-{uuid4()}")
    finally:
        await db.disconnect()


async def start(store: Store):
    identity = uuid4().hex[:26]
    return await store.start(
        Admission(
            request_id=identity,
            trace_id=f"trace-{identity}",
            project_id="project",
            customer_model="model",
            mode="active",
        )
    )


async def test_replay_refused_before_provider_can_be_called(store: Store) -> None:
    row = await start(store)
    with pytest.raises(RuntimeError, match="replay refused"):
        await store.start(
            Admission(
                request_id=row.id,
                trace_id=row.trace_id,
                project_id=row.project_id,
                customer_model=row.customer_model,
                mode=row.mode,
            )
        )
    assert await store.check_version()


async def test_attempt_closure_and_outbox_are_transactional(store: Store) -> None:
    row = await start(store)
    attempt = await store.begin_attempt(
        row,
        attempt_id=str(uuid4()),
        model="provider/model",
        group="model",
        deployment="deployment",
        provider="provider",
    )
    body = Attempt(
        type="attempt.finished",
        attemptId=attempt.id,
        logId=attempt.log_id,
        ordinal=1,
        kind="primary",
        model=attempt.model,
        modelGroup=attempt.model_group,
        deploymentId=attempt.deployment_id,
        provider=attempt.provider,
        startedAt=attempt.started_at,
        endedAt=attempt.started_at,
        outcome="success",
        usage=KnownUsage(state="known", promptTokens=100, completionTokens=20, cacheReadTokens=10),
    )
    await store.finish(row, attempt, body)
    await store.close(row, "success")
    events = await store.database.writer_db.query_raw(
        "SELECT payload FROM litellm_usage.outbox WHERE request_id=$1 ORDER BY id", row.id
    )
    parsed = TypeAdapter(list[dict[str, object]]).validate_python(events)
    batch = UsageBatch.model_validate({"events": [item["payload"] for item in parsed]})
    assert [e.body.type for e in batch.events] == ["request.started", "attempt.finished", "request.closed"]
    assert all(e.digest == e.content_digest() for e in batch.events)
    with pytest.raises(RuntimeError, match="closed"):
        await store.begin_attempt(
            row, attempt_id=str(uuid4()), model="provider/model", group="model", deployment=None, provider=None
        )


async def test_multiworker_claims_are_disjoint_and_expire_for_takeover(store: Store) -> None:
    row = await start(store)
    other = Store(store.database, f"other-{uuid4()}")
    first, second = await asyncio.gather(store.claim(), other.claim())
    assert not {d.id for d in first} & {d.id for d in second}
    await store.database.writer_db.execute_raw(
        "UPDATE litellm_usage.outbox SET lease_until=now()-interval '1 second' WHERE request_id=$1", row.id
    )
    reclaimed = await other.claim()
    assert any(d.payload.requestId == row.id for d in reclaimed)


async def test_ack_loss_retries_and_partial_ack_never_drops_an_event(store: Store) -> None:
    row = await start(store)
    seen: list[str] = []

    def receiver(request: httpx.Request) -> httpx.Response:
        events = UsageBatch.model_validate_json(request.content).events
        seen.extend(event.eventId for event in events)
        return httpx.Response(
            200,
            json={
                "acknowledgements": [
                    {"eventId": event.eventId, "version": event.version, "digest": event.digest, "status": "accepted"}
                    for event in events
                    if event.requestId == row.id
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(receiver)) as client:
        await deliver(store, client, "http://receiver/usage", "test-token")
    state = await store.database.writer_db.query_raw(
        "SELECT disposition FROM litellm_usage.outbox WHERE request_id=$1", row.id
    )
    assert state == [{"disposition": "delivered"}]
    assert f"{row.id}/started" in seen


async def test_unknown_can_be_enriched_but_known_usage_is_not_overwritten(store: Store) -> None:
    row = await start(store)
    attempt = await store.begin_attempt(
        row, attempt_id=str(uuid4()), model="provider/model", group="model", deployment=None, provider=None
    )
    unknown = Attempt(
        type="attempt.finished",
        attemptId=attempt.id,
        logId=attempt.log_id,
        ordinal=1,
        kind="primary",
        model=attempt.model,
        modelGroup=attempt.model_group,
        deploymentId=None,
        provider=None,
        startedAt=attempt.started_at,
        endedAt=attempt.started_at,
        outcome="failure",
        usage=UnknownUsage(state="unknown", reason="process_lost"),
    )
    known = unknown.model_copy(
        update={"usage": KnownUsage(state="known", promptTokens=1, completionTokens=2, cacheReadTokens=0)}
    )
    await store.finish(row, attempt, unknown)
    await store.finish(row, attempt, known)
    events = await store.database.writer_db.query_raw(
        "SELECT version FROM litellm_usage.outbox WHERE event_id=$1 ORDER BY version", f"{row.id}/attempt/{attempt.id}"
    )
    assert events == [{"version": 1}, {"version": 2}]


def test_contract_samples_have_valid_digest_and_match_schema() -> None:
    root = Path(__file__).resolve().parents[4] / "litellm/proxy/usage_journal/contracts"
    assert json.loads((root / "batch.schema.json").read_text()) == UsageBatch.model_json_schema()
    batch = UsageBatch.model_validate_json((root / "examples.json").read_text())
    assert all(event.digest == event.content_digest() for event in batch.events)


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 1}, {"prompt_tokens": -1, "completion_tokens": 2}])
def test_missing_or_invalid_usage_remains_unknown(usage: object) -> None:
    assert isinstance(usage_from({"usage": usage}), UnknownUsage)


def test_anthropic_stream_needs_terminal_evidence_and_merges_usage() -> None:
    wire = WireUsage()
    wire.feed(
        'data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0,"cache_read_input_tokens":5}}}\n\n'
    )
    assert isinstance(wire.result(), UnknownUsage)
    wire.feed('data: {"type":"message_delta","usage":{"output_tokens":7}}\n\n')
    assert isinstance(wire.result(), UnknownUsage)
    wire.feed('data: {"type":"message_stop"}\n\n')
    assert wire.result() == KnownUsage(state="known", promptTokens=15, completionTokens=7, cacheReadTokens=5)


async def test_actual_async_client_lifecycle_writes_attempt_and_closed_membership(store: Store, monkeypatch) -> None:
    import litellm
    from starlette.requests import Request
    from litellm.proxy.usage_journal.lifecycle import configure, durable_request

    identity = uuid4().hex[:26]
    monkeypatch.setenv("LITELLM_USAGE_REPORT_TOKEN", "journal-test-token")
    headers = {
        "x-guangrouter-usage-protocol": "litellm-usage/1",
        "x-guangrouter-usage-request": identity,
        "x-idempotency-key": f"trace-{identity}",
        "x-project-id": "project",
        "x-guangrouter-customer-model": "model",
        "x-guangrouter-usage-mode": "active",
        "x-guangrouter-usage-token": "journal-test-token",
    }
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
        }
    )

    @durable_request
    async def route(*, request: Request) -> object:
        return await litellm.acompletion(
            model="openai/test-model",
            api_key="sk-unit",
            mock_response="hello",
            messages=[{"role": "user", "content": "hello"}],
        )

    configure(store)
    try:
        await route(request=request)
    finally:
        configure(None)
    events = TypeAdapter(list[dict[str, object]]).validate_python(
        await store.database.writer_db.query_raw(
            "SELECT payload FROM litellm_usage.outbox WHERE request_id=$1 ORDER BY id", identity
        )
    )
    batch = UsageBatch.model_validate({"events": [item["payload"] for item in events]})
    assert [e.body.type for e in batch.events] == ["request.started", "attempt.finished", "request.closed"]
    assert isinstance(batch.events[1].body, Attempt)
    assert batch.events[1].body.usage.state == "known"
    assert batch.events[2].body.attemptIds == [batch.events[1].body.attemptId]
