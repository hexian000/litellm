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
from litellm.proxy.usage_journal.protocol import Attempt, KnownUsage, UnknownUsage, UsageBatch, contract_digest
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
    instance = f"test-{uuid4()}"
    try:
        yield Store(Database(db), instance)
    finally:
        owned = TypeAdapter(list[dict[str, str]]).validate_python(await db.query_raw(
            "SELECT request_id FROM litellm_usage.outbox WHERE payload->>'instanceId'=$1 "
            "AND event_id=request_id||'/started'", instance))
        for item in owned:
            request_id = item["request_id"]
            await db.execute_raw("DELETE FROM litellm_usage.recovery WHERE attempt_id IN "
                                 "(SELECT id FROM litellm_usage.attempt WHERE request_id=$1)", request_id)
            await db.execute_raw("DELETE FROM litellm_usage.attempt WHERE request_id=$1", request_id)
            await db.execute_raw("DELETE FROM litellm_usage.outbox WHERE request_id=$1", request_id)
            await db.execute_raw("DELETE FROM litellm_usage.request WHERE id=$1", request_id)
        await db.execute_raw("""
            UPDATE litellm_usage.control SET
              pending=(SELECT count(*) FROM litellm_usage.outbox WHERE disposition!='delivered'),
              open_requests=(SELECT count(*) FROM litellm_usage.request WHERE closed_at IS NULL),
              unknown_attempts=(SELECT count(*) FROM litellm_usage.attempt WHERE NOT usage_known)
            WHERE id=1
        """)
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
        endedAt=None,
        outcome="unknown",
        usage=UnknownUsage(state="unknown", reason="process_lost"),
    )
    known = unknown.model_copy(
        update={"usage": KnownUsage(state="known", promptTokens=1, completionTokens=2, cacheReadTokens=0),
                "outcome": "success", "endedAt": attempt.started_at}
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
        "x-guangrouter-usage-contract": contract_digest(),
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


async def test_closed_request_missing_attempt_is_recovered_from_durable_source(store: Store) -> None:
    from litellm.proxy.usage_journal.evidence import recover_evidence

    row = await start(store)
    attempt = await store.begin_attempt(
        row,
        attempt_id=str(uuid4()),
        model="provider/model",
        group="model",
        deployment="deployment",
        provider="provider",
    )
    await store.close(row, "success")
    await store.retain_source(
        attempt,
        {
            "request_id": attempt.log_id,
            "endTime": attempt.started_at.isoformat(),
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "model_id": "deployment",
            "spend": "0.000123",
            "status": "success",
            "call_type": "acompletion",
        },
    )
    other = Store(store.database, f"replacement-{uuid4()}")
    await recover_evidence(other)
    result = await store.database.writer_db.query_raw(
        "SELECT usage_known,body FROM litellm_usage.attempt WHERE id=$1", attempt.id
    )
    assert result[0]["usage_known"] is True
    assert result[0]["body"]["usage"]["promptTokens"] == 10
    recovery = await store.database.writer_db.query_raw(
        "SELECT completed,observation FROM litellm_usage.recovery WHERE attempt_id=$1", attempt.id
    )
    assert recovery[0]["completed"] is True
    assert recovery[0]["observation"]["cost"]["amount"] == "0.000123"
    events = await store.database.writer_db.query_raw(
        "SELECT event_id FROM litellm_usage.outbox WHERE request_id=$1", row.id
    )
    assert {r["event_id"] for r in events} == {
        f"{row.id}/started",
        f"{row.id}/closed",
        f"{row.id}/attempt/{attempt.id}",
        f"{row.id}/observation/{attempt.id}",
    }


@pytest.mark.parametrize("source_first", [False, True])
async def test_stale_native_source_preserves_the_first_committed_completion(store: Store, source_first: bool) -> None:
    from datetime import timedelta

    from litellm.proxy.usage_journal.evidence import apply_source

    row = await start(store)
    attempt = await store.begin_attempt(
        row, attempt_id=str(uuid4()), model="provider/model", group="model", deployment=None, provider=None
    )
    finished = Attempt(
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
        endedAt=attempt.started_at + timedelta(milliseconds=2),
        outcome="success",
        supplierAttempted=True,
        usage=KnownUsage(state="known", promptTokens=7, completionTokens=1, cacheReadTokens=0),
    )
    source = {
        "request_id": attempt.log_id,
        "endTime": (attempt.started_at + timedelta(milliseconds=1)).isoformat(),
        "prompt_tokens": 7,
        "completion_tokens": 1,
        "spend": "0.000009",
        "status": "success",
        "call_type": "acompletion",
    }
    await store.close(row, "success")
    if source_first:
        await apply_source(store, attempt, source)
        await store.finish(row, attempt, finished)
    else:
        await store.finish(row, attempt, finished)
        await apply_source(store, attempt, source)
    await apply_source(store, attempt, source)
    conflicts = await store.database.writer_db.query_raw(
        "SELECT conflicted FROM litellm_usage.request WHERE id=$1", row.id
    )
    assert conflicts == [{"conflicted": False}]
    events = UsageBatch.model_validate(
        {
            "events": [
                r["payload"]
                for r in await store.database.writer_db.query_raw(
                    "SELECT payload FROM litellm_usage.outbox WHERE event_id=$1 ORDER BY version",
                    f"{row.id}/attempt/{attempt.id}",
                )
            ]
        }
    ).events
    assert len(events) == 1
    assert isinstance(events[0].body, Attempt)
    expected = attempt.started_at + timedelta(milliseconds=1 if source_first else 2)
    assert events[0].body.endedAt == expected
    observation = await store.database.writer_db.query_raw(
        "SELECT observation FROM litellm_usage.recovery WHERE attempt_id=$1", attempt.id
    )
    assert observation[0]["observation"]["endedAt"] == events[0].body.model_dump(mode="json")["endedAt"]


async def test_stale_native_source_retains_contradictory_known_usage(store: Store) -> None:
    from datetime import timedelta

    from litellm.proxy.usage_journal.evidence import apply_source

    row = await start(store)
    attempt = await store.begin_attempt(
        row, attempt_id=str(uuid4()), model="provider/model", group="model", deployment=None, provider=None
    )
    finished = Attempt(
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
        endedAt=attempt.started_at + timedelta(milliseconds=2),
        outcome="success",
        supplierAttempted=True,
        usage=KnownUsage(state="known", promptTokens=7, completionTokens=1, cacheReadTokens=0),
    )
    await store.finish(row, attempt, finished)
    await store.close(row, "success")
    await apply_source(
        store,
        attempt,
        {
            "request_id": attempt.log_id,
            "endTime": (attempt.started_at + timedelta(milliseconds=1)).isoformat(),
            "prompt_tokens": 8,
            "completion_tokens": 1,
            "spend": "0.000009",
            "status": "success",
            "call_type": "acompletion",
        },
    )
    assert await store.database.writer_db.query_raw(
        "SELECT conflicted FROM litellm_usage.request WHERE id=$1", row.id
    ) == [{"conflicted": True}]
    versions = await store.database.writer_db.query_raw(
        "SELECT version FROM litellm_usage.outbox WHERE event_id=$1 ORDER BY version", f"{row.id}/attempt/{attempt.id}"
    )
    assert versions == [{"version": 1}, {"version": 2}]


async def test_native_spend_capture_retains_source_before_log_delivery(store: Store) -> None:
    from litellm.proxy.usage_journal.evidence import capture, recover_evidence

    row = await start(store)
    attempt = await store.begin_attempt(
        row, attempt_id=str(uuid4()), model="provider/model", group="model", deployment=None, provider=None
    )
    await capture(store, {
        "request_id": attempt.log_id,
        "endTime": attempt.started_at,
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "spend": 0,
        "status": "success",
        "call_type": "acompletion",
        "metadata": json.dumps({
            "usage_object": {"prompt_tokens": 10, "completion_tokens": 3},
            "additional_usage_values": {"journal_cost_known": True},
        }),
    })
    retained = await store.database.writer_db.query_raw(
        "SELECT source_payload FROM litellm_usage.recovery WHERE attempt_id=$1", attempt.id
    )
    assert retained[0]["source_payload"]["metadata"]["usage_object"]["prompt_tokens"] == 10
    await store.close(row, "success")
    await recover_evidence(store)
    recovered = await store.database.writer_db.query_raw(
        "SELECT completed,observation FROM litellm_usage.recovery WHERE attempt_id=$1", attempt.id
    )
    assert recovered[0]["completed"] is True
    assert recovered[0]["observation"]["cost"]["amount"] == "0.000000"


async def test_supplier_failure_is_observed_without_inventing_usage(store: Store) -> None:
    from litellm.proxy.usage_journal.evidence import apply_source

    row = await start(store)
    attempt = await store.begin_attempt(
        row, attempt_id=str(uuid4()), model="provider/model", group="model", deployment=None, provider=None
    )
    await store.close(row, "failure")
    await apply_source(store, attempt, {
        "request_id": attempt.log_id,
        "endTime": attempt.started_at,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "spend": None,
        "status": "failure",
        "status_code": 503,
        "call_type": "acompletion",
    })
    recovered = await store.database.writer_db.query_raw(
        "SELECT completed,observation FROM litellm_usage.recovery WHERE attempt_id=$1", attempt.id
    )
    assert recovered[0]["completed"] is False
    assert recovered[0]["observation"]["outcome"] == "failure"
    assert recovered[0]["observation"]["statusCode"] == 503
    assert recovered[0]["observation"]["cost"]["state"] == "unknown"
    usage = await store.database.writer_db.query_raw(
        "SELECT usage_known,body FROM litellm_usage.attempt WHERE id=$1", attempt.id
    )
    assert usage[0]["usage_known"] is False
    assert usage[0]["body"]["usage"]["state"] == "unknown"
    await apply_source(store, attempt, {
        "request_id": attempt.log_id,
        "endTime": attempt.started_at,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "spend": None,
        "status": "failure",
        "status_code": 503,
        "call_type": "acompletion",
    })
    assert await store.database.writer_db.query_raw(
        "SELECT version FROM litellm_usage.outbox WHERE event_id=$1", f"{row.id}/attempt/{attempt.id}"
    ) == [{"version": 1}]


async def test_actual_client_failure_retains_attempt_source_before_context_unwinds(store: Store) -> None:
    import litellm
    from openai import AsyncOpenAI

    from litellm.proxy.usage_journal.evidence import recover_evidence
    from litellm.proxy.usage_journal.lifecycle import RequestHandle, configure, current_request

    def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "unavailable", "type": "server_error"}})

    row = await start(store)
    token = current_request.set(RequestHandle(store, row))
    configure(store)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as http_client:
            async with AsyncOpenAI(api_key="sk-unit", max_retries=0, http_client=http_client) as client:
                with pytest.raises(litellm.ServiceUnavailableError):
                    await litellm.acompletion(
                        model="openai/test-model", client=client, messages=[{"role": "user", "content": "hello"}]
                    )
    finally:
        configure(None)
        current_request.reset(token)
    await store.close(row, "failure")
    await recover_evidence(store)
    source = await store.database.writer_db.query_raw(
        "SELECT c.source_payload,c.observation,a.usage_known FROM litellm_usage.recovery c "
        "JOIN litellm_usage.attempt a ON a.id=c.attempt_id WHERE a.request_id=$1", row.id
    )
    assert source[0]["source_payload"]["request_id"] == source[0]["observation"]["logId"]
    assert source[0]["observation"]["outcome"] == "failure"
    assert source[0]["observation"]["statusCode"] == 503
    assert source[0]["observation"]["cost"]["state"] == "unknown"
    assert source[0]["usage_known"] is False


async def test_recovery_without_evidence_keeps_unknown_and_fences_old_owner(store: Store) -> None:
    from litellm.proxy.usage_journal.runtime import recover

    row = await start(store)
    attempt = await store.begin_attempt(
        row, attempt_id=str(uuid4()), model="provider/model", group="model", deployment=None, provider=None
    )
    await store.database.writer_db.execute_raw(
        "UPDATE litellm_usage.request SET started_at=now()-interval '2 minutes' WHERE id=$1", row.id
    )
    other = Store(store.database, f"replacement-{uuid4()}")
    await recover(other)
    found = await store.database.writer_db.query_raw(
        "SELECT usage_known,body FROM litellm_usage.attempt WHERE id=$1", attempt.id
    )
    assert found[0]["usage_known"] is False
    assert found[0]["body"]["endedAt"] is None
    assert found[0]["body"]["outcome"] == "unknown"
    with pytest.raises(RuntimeError, match="closed"):
        await store.begin_attempt(
            row, attempt_id=str(uuid4()), model="provider/model", group="model", deployment=None, provider=None
        )


async def test_recovery_claims_are_bounded_and_disjoint(store: Store) -> None:
    row = await start(store)
    attempt = await store.begin_attempt(
        row, attempt_id=str(uuid4()), model="provider/model", group="model", deployment=None, provider=None
    )
    await store.close(row, "success")
    other = Store(store.database, f"replacement-{uuid4()}")
    first, second = await asyncio.gather(store.recovery_candidates(), other.recovery_candidates())
    assert len(first) <= 100 and len(second) <= 100
    assert not {a.id for a in first} & {a.id for a in second}
    assert attempt.id in {a.id for a in (*first, *second)}


async def test_recovery_resumes_after_takeover_without_closing_live_requests(store: Store) -> None:
    from litellm.proxy.usage_journal.runtime import recover

    row = await start(store)
    attempt = await store.begin_attempt(
        row, attempt_id=str(uuid4()), model="provider/model", group="model", deployment=None, provider=None
    )
    await store.database.writer_db.execute_raw(
        "UPDATE litellm_usage.request SET started_at=now()-interval '2 minutes' WHERE id=$1", row.id
    )
    other = Store(store.database, f"replacement-{uuid4()}")
    await other.heartbeat(10 * 1024**3)
    adopted = await other.orphaned()
    assert row.id in {request.id for request in adopted}
    live = await start(other)

    await recover(other)

    recovered = await store.database.writer_db.query_raw(
        "SELECT closed_at,outcome FROM litellm_usage.request WHERE id=$1", row.id
    )
    assert recovered[0]["closed_at"] is not None
    assert recovered[0]["outcome"] == "unknown"
    pending = await store.database.writer_db.query_raw(
        "SELECT usage_known,body FROM litellm_usage.attempt WHERE id=$1", attempt.id
    )
    assert pending[0]["usage_known"] is False
    assert pending[0]["body"]["endedAt"] is None
    assert await store.database.writer_db.query_raw(
        "SELECT closed_at FROM litellm_usage.request WHERE id=$1", live.id
    ) == [{"closed_at": None}]


@pytest.mark.parametrize(
    "response,expected",
    [
        ({"usage": {"prompt_tokens": 10, "completion_tokens": 2}}, (10, 2, 0)),
        ({"usage": {"prompt_tokens": 10, "completion_tokens": 0}}, (10, 0, 0)),
        (
            {
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.1]}],
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
            (10, 0, 0),
        ),
        ({"usage": {"input_tokens": 10, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 3}}}, (10, 2, 3)),
        ({"usage": {"input_tokens": 10, "output_tokens": 2, "cache_creation_input_tokens": 4}}, (14, 2, 0)),
    ],
)
def test_five_inference_usage_shapes_preserve_actual_counts(response, expected):
    usage = usage_from(response)
    assert isinstance(usage, KnownUsage)
    assert (usage.promptTokens, usage.completionTokens, usage.cacheReadTokens) == expected


def test_stream_usage_needs_completion_and_accepts_fragmented_crlf():
    wire = WireUsage()
    wire.feed('data: {"usage":{"prompt_tokens":10,"completion_tokens":2}}\r')
    wire.feed("\n\r\n")
    assert isinstance(wire.result(), UnknownUsage)
    wire.feed("data: [DONE]\r\n\r\n")
    assert wire.result() == KnownUsage(state="known", promptTokens=10, completionTokens=2, cacheReadTokens=0)


async def test_delivery_loss_and_partial_ack_leave_only_missing_events_pending(store):
    row = await start(store)
    await store.close(row, "success")

    def lost(request):
        raise httpx.ReadTimeout("ack lost", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lost)) as client:
        await deliver(store, client, "http://receiver/usage", "test-token")
    states = await store.database.writer_db.query_raw(
        "SELECT disposition FROM litellm_usage.outbox WHERE request_id=$1", row.id
    )
    assert [s["disposition"] for s in states] == ["pending", "pending"]
    await store.database.writer_db.execute_raw(
        "UPDATE litellm_usage.outbox SET next_attempt_at=now() WHERE request_id=$1", row.id
    )

    def partial(request):
        batch = UsageBatch.model_validate_json(request.content)
        assert request.headers["X-GuangRouter-Usage-Contract"] == contract_digest()
        return httpx.Response(
            200,
            json={
                "acknowledgements": [
                    {"eventId": e.eventId, "version": e.version, "digest": e.digest, "status": "accepted"}
                    for e in batch.events
                    if e.requestId == row.id and e.body.type == "request.started"
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(partial)) as client:
        await deliver(store, client, "http://receiver/usage", "test-token")
    states = await store.database.writer_db.query_raw(
        "SELECT event_id,disposition FROM litellm_usage.outbox WHERE request_id=$1 ORDER BY id", row.id
    )
    assert states == [
        {"event_id": f"{row.id}/started", "disposition": "delivered"},
        {"event_id": f"{row.id}/closed", "disposition": "pending"},
    ]


async def test_incremental_feature_migration_preserves_original_journal(store, monkeypatch):
    from urllib.parse import urlsplit, urlunsplit

    from litellm.proxy.usage_journal.migrate import migrate

    parts = urlsplit(os.environ["LITELLM_USAGE_TEST_DATABASE_URL"])
    if parts.hostname not in {"127.0.0.1", "localhost"}:
        pytest.skip("migration proof requires disposable loopback PostgreSQL")
    name = f"usage_upgrade_{uuid4().hex}"
    await store.database.writer_db.execute_raw(f'CREATE DATABASE "{name}"')
    url = urlunsplit(parts._replace(path=f"/{name}"))
    db = Prisma(datasource={"url": url})
    await db.connect()
    try:
        root = Path(__file__).resolve().parents[4] / "litellm/proxy/usage_journal"
        for statement in (root / "migrations/001.sql").read_text().split(";"):
            if statement.strip():
                await db.execute_raw(statement)
        await db.execute_raw("INSERT INTO litellm_usage.revision(version,operation_id) VALUES(1,'old-source')")
        await db.execute_raw("""INSERT INTO litellm_usage.request
            (id,trace_id,project_id,customer_model,mode,instance_id)
            VALUES('legacy','legacy','project','model','active','old')""")
        await db.execute_raw("""INSERT INTO litellm_usage.attempt
            (id,request_id,ordinal,log_id,model,model_group,kind)
            VALUES('legacy-attempt','legacy',1,'legacy-log','provider/model','model','primary')""")
        monkeypatch.setenv("DATABASE_URL", url)
        await migrate(name, False)
        assert await db.query_raw("SELECT to_regclass('litellm_usage.recovery')::text AS name") == [{"name": None}]
        await migrate(name, True)
        await migrate(name, True)
        assert await db.query_raw("SELECT version FROM litellm_usage.revision") == [{"version": 1}]
        assert await db.query_raw("SELECT version FROM litellm_usage.feature_revision") == [{"version": 2}]
        assert await db.query_raw("SELECT attempt_id,completed FROM litellm_usage.recovery") == [
            {"attempt_id": "legacy-attempt", "completed": False}
        ]
        await db.execute_raw("ALTER TABLE litellm_usage.recovery ADD COLUMN unexpected text")
        with pytest.raises(RuntimeError, match="unexpected recovery column"):
            await migrate(name, False)
    finally:
        await db.disconnect()
        await store.database.writer_db.execute_raw(f'DROP DATABASE "{name}" WITH (FORCE)')


@pytest.mark.parametrize("mode,suppressed", [("active", True), ("shadow", False)])
async def test_only_active_journal_owns_the_legacy_business_callback(store, mode, suppressed):
    from datetime import UTC, datetime

    from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger
    from litellm.proxy.usage_journal.lifecycle import RequestHandle, current_request

    row = await start(store)
    row = row.model_copy(update={"mode": mode})
    logger = GenericAPILogger(
        endpoint="http://receiver/events", callback_name="workbench_model_control", batch_size=100
    )
    token = current_request.set(RequestHandle(store, row))
    try:
        await logger.async_log_success_event(
            {"standard_logging_object": {"id": "test"}}, {}, datetime.now(UTC), datetime.now(UTC)
        )
        assert bool(logger.log_queue) is not suppressed
    finally:
        current_request.reset(token)


@pytest.mark.parametrize("journaled", [False, True])
@pytest.mark.parametrize("cost", [None, 0, 0.125])
def test_cache_hit_keeps_journal_log_identity_and_preserves_legacy_suffix(journaled, cost):
    from datetime import UTC, datetime

    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
    from litellm.proxy.usage_journal.lifecycle import AttemptHandle, current_attempt
    from litellm.proxy.usage_journal.store import AttemptRow

    now = datetime.now(UTC)
    attempt = AttemptHandle(AttemptRow(
        id="cache-attempt", request_id="cache-request", ordinal=1, log_id="cache-attempt",
        model="provider/model", model_group="model", kind="primary", started_at=now,
    ))
    token = current_attempt.set(attempt if journaled else None)
    try:
        payload = get_logging_payload(
            kwargs={"model": "provider/model", "cache_hit": True, "litellm_call_id": "cache-attempt",
                    **({"response_cost": cost} if cost is not None else {})},
            response_obj={"id": "cached-response", "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            start_time=now, end_time=now,
        )
    finally:
        current_attempt.reset(token)
    metadata = json.loads(payload["metadata"])["additional_usage_values"]
    if journaled:
        assert payload["request_id"] == "cache-attempt"
        assert metadata["journal_cost_known"] is (cost is not None)
    else:
        assert "_cache_hit" in payload["request_id"]
        assert "journal_cost_known" not in metadata


@pytest.mark.parametrize("digest", [None, "0" * 64])
def test_admission_rejects_missing_or_mismatched_contract(digest, monkeypatch):
    from fastapi import HTTPException
    from starlette.requests import Request

    from litellm.proxy.usage_journal.runtime import authorize

    monkeypatch.setenv("LITELLM_USAGE_REPORT_TOKEN", "test-token")
    headers = [(b"x-guangrouter-usage-token", b"test-token"),
               (b"x-guangrouter-usage-protocol", b"litellm-usage/1")]
    if digest is not None:
        headers.append((b"x-guangrouter-usage-contract", digest.encode()))
    request = Request({"type": "http", "headers": headers})
    with pytest.raises(HTTPException) as failed:
        authorize(request)
    assert failed.value.status_code == 409


@pytest.mark.parametrize("cost,proven,known", [(None, False, False), (0, False, False), (0, None, False), (0, True, True), ("0.125", None, True)])
def test_missing_cost_defaults_cannot_become_proven_zero(cost, proven, known):
    from litellm.proxy.usage_journal.evidence import cost_from
    from litellm.proxy.usage_journal.protocol import KnownCost

    result = cost_from({"spend": cost, "metadata": {"additional_usage_values": {"journal_cost_known": proven}}})
    assert isinstance(result, KnownCost) is known
