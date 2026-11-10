from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal, Protocol

from pydantic import BaseModel, TypeAdapter

from ..db.prisma_batch import ROWS, TIMEOUTS, Atomic, Rows, Statement
from .protocol import PROTOCOL, Attempt, Closed, KnownCost, Observation, Started, UsageEvent

logger = logging.getLogger(__name__)
Scalar = str | int | bool | None


class Sql(Protocol):
    async def query_raw(self, query: str, *args: Scalar) -> object: ...

    async def execute_raw(self, query: str, *args: Scalar) -> object: ...


class Transaction(Protocol):
    async def __aenter__(self) -> Sql: ...

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> object: ...


class Database(Protocol):
    def tx(self) -> Transaction: ...

    @property
    def writer_db(self) -> Sql: ...


async def _query_sequence(tx: Sql, statements: tuple[Statement, ...]) -> tuple[Rows, ...]:
    if not statements:
        return ()
    first = ROWS.validate_python(await tx.query_raw(statements[0].query, *statements[0].parameters))
    remaining = await _query_sequence(tx, statements[1:])
    return (first, *remaining)


class Admission(BaseModel):
    request_id: str
    trace_id: str
    project_id: str
    customer_model: str
    mode: str


class RequestRow(BaseModel):
    id: str
    trace_id: str
    project_id: str
    customer_model: str
    mode: str
    instance_id: str
    started_at: datetime
    closed_at: datetime | None = None


class AttemptRow(BaseModel):
    id: str
    request_id: str
    ordinal: int
    log_id: str
    model: str
    model_group: str
    deployment_id: str | None = None
    provider: str | None = None
    kind: Literal["primary", "retry", "fallback"]
    started_at: datetime
    version: int = 0
    usage_known: bool = False
    body: Attempt | None = None


class Delivery(BaseModel):
    id: int
    payload: UsageEvent
    attempts: int


class Revision(BaseModel):
    version: int
    name: str | None


REQUESTS = TypeAdapter(Sequence[RequestRow])
ATTEMPTS = TypeAdapter(Sequence[AttemptRow])
DELIVERIES = TypeAdapter(Sequence[Delivery])
REVISIONS = TypeAdapter(tuple[Revision, ...])
VERSIONS = TypeAdapter(tuple[Mapping[str, int], ...])


def _retain_native_first_response(previous: Observation | None, body: Observation) -> Observation:
    if previous is None or previous.firstResponseAt is None or body.firstResponseAt is None:
        return body
    earlier, incoming = previous.firstResponseAt, body.firstResponseAt
    if earlier == incoming:
        return body
    if (
        earlier.replace(microsecond=earlier.microsecond // 1000 * 1000) != incoming
        and incoming.replace(microsecond=incoming.microsecond // 1000 * 1000) != earlier
    ):
        return body
    return body.model_copy(update=MappingProxyType({"firstResponseAt": earlier}))


class DeliveryResult(BaseModel):
    id: int
    status: str
    reason: str = ""
    delay_ms: int = 100


class Store:
    def __init__(
        self, database: Database, instance_id: str, max_pending: int = 100000, *, atomic: Atomic | None = None
    ) -> None:
        self.database = database
        self.instance_id = instance_id
        self.max_pending = max_pending
        self._transactions = asyncio.Semaphore(4)
        self._atomic = atomic

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Sql]:
        async with self._transactions, self.database.tx() as tx:
            await tx.query_raw(TIMEOUTS.query)
            yield tx

    async def query_transaction(self, *statements: Statement) -> tuple[Rows, ...]:
        async with self._transactions:
            if self._atomic is not None:
                return await self._atomic(statements)
            async with self.database.tx() as tx:
                await tx.query_raw(TIMEOUTS.query)
                return await _query_sequence(tx, statements)

    async def check_version(self) -> bool:
        rows = REVISIONS.validate_python(
            await self.database.writer_db.query_raw(
                "SELECT version,to_regclass('litellm_usage.feature_revision')::text AS name FROM litellm_usage.revision"
            )
        )
        if tuple(row.version for row in rows) != (1,):
            return False
        if not rows[0].name:
            return False
        versions = await self.database.writer_db.query_raw("SELECT version FROM litellm_usage.feature_revision")
        return tuple(row["version"] for row in VERSIONS.validate_python(versions)) == (2,)

    def event(
        self, request: RequestRow, body: Started | Attempt | Closed | Observation, version: int = 1
    ) -> UsageEvent:
        suffix = (
            f"attempt/{body.attemptId}"
            if isinstance(body, Attempt)
            else f"observation/{body.attemptId}"
            if isinstance(body, Observation)
            else body.type.removeprefix("request.")
        )
        now = datetime.now(UTC)
        parsed = UsageEvent.model_validate(
            MappingProxyType(
                {
                    "protocol": PROTOCOL,
                    "eventId": f"{request.id}/{suffix}",
                    "version": version,
                    "occurredAt": now,
                    "instanceId": self.instance_id,
                    "requestId": request.id,
                    "traceId": request.trace_id,
                    "projectId": request.project_id,
                    "customerModel": request.customer_model,
                    "mode": request.mode,
                    "body": body.model_dump(mode="json"),
                    "digest": "0" * 64,
                }
            )
        )
        return parsed.model_copy(update=MappingProxyType({"digest": parsed.content_digest()}))

    async def emit(
        self, tx: Sql, request: RequestRow, body: Started | Attempt | Closed | Observation, version: int = 1
    ) -> None:
        event = self.event(request, body, version)
        await tx.execute_raw(
            """
            WITH emitted AS (
                INSERT INTO litellm_usage.outbox(event_id, version, request_id, payload, digest)
                VALUES ($1, $2, $3, $4::jsonb, $5) RETURNING id
            )
            UPDATE litellm_usage.control SET pending=pending+(SELECT count(*) FROM emitted) WHERE id=1
            """,
            event.eventId,
            version,
            request.id,
            event.model_dump_json(),
            event.digest,
        )

    async def start(self, admission: Admission) -> RequestRow:
        async with self.transaction() as tx:
            request = RequestRow(
                id=admission.request_id,
                trace_id=admission.trace_id,
                project_id=admission.project_id,
                customer_model=admission.customer_model,
                mode=admission.mode,
                instance_id=self.instance_id,
                started_at=datetime.now(UTC),
            )
            event = self.event(request, Started(type="request.started", startedAt=request.started_at))
            rows = await tx.query_raw(
                """
                WITH capacity AS (
                    SELECT id FROM litellm_usage.control
                    WHERE id=1 AND admitted AND pending+open_requests+unknown_attempts < $8 FOR UPDATE
                ), inserted AS (
                    INSERT INTO litellm_usage.request(
                        id, trace_id, project_id, customer_model, mode, instance_id, started_at)
                    SELECT $1,$2,$3,$4,$5,$6,$7::text::timestamptz FROM capacity
                    ON CONFLICT DO NOTHING RETURNING *
                ), emitted AS (
                    INSERT INTO litellm_usage.outbox(event_id, version, request_id, payload, digest)
                    SELECT $9,1,id,$10::jsonb,$11 FROM inserted RETURNING id
                ), counted AS (
                    UPDATE litellm_usage.control SET open_requests=open_requests+1,pending=pending+1
                    WHERE id=1 AND EXISTS(SELECT 1 FROM emitted) RETURNING id
                )
                SELECT inserted.* FROM inserted CROSS JOIN counted
                """,
                request.id,
                request.trace_id,
                request.project_id,
                request.customer_model,
                request.mode,
                self.instance_id,
                request.started_at.isoformat(),
                self.max_pending,
                event.eventId,
                event.model_dump_json(),
                event.digest,
            )
            requests = REQUESTS.validate_python(rows)
            if requests:
                return requests[0]
            existing = await tx.query_raw("SELECT id FROM litellm_usage.request WHERE id=$1", request.id)
            if existing:
                raise RuntimeError("request journal already exists; provider replay refused")
            raise RuntimeError("usage journal capacity exhausted")

    async def begin_attempt(
        self,
        request: RequestRow,
        *,
        attempt_id: str,
        model: str,
        group: str,
        deployment: str | None,
        provider: str | None,
    ) -> AttemptRow:
        open_rows, rows = await self.query_transaction(
            Statement(
                query="SELECT id FROM litellm_usage.request WHERE id=$1 AND closed_at IS NULL "
                "AND instance_id=$2 FOR NO KEY UPDATE",
                parameters=(request.id, self.instance_id),
            ),
            Statement(
                query="""
                WITH capacity AS (
                    SELECT id FROM litellm_usage.control
                    WHERE id=1 AND admitted AND pending+open_requests+unknown_attempts < $7 FOR UPDATE
                ), inserted AS (
                    INSERT INTO litellm_usage.attempt(id, request_id, ordinal, log_id, model, model_group,
                                                      deployment_id, provider, kind)
                    SELECT $1,$2,count(*)::integer+1,$1,$3,$4,$5,$6,
                        CASE WHEN count(*)=0 THEN 'primary'
                             WHEN bool_and(model=$3) THEN 'retry' ELSE 'fallback' END
                    FROM litellm_usage.attempt WHERE request_id=$2
                    HAVING EXISTS(SELECT 1 FROM capacity)
                        AND EXISTS(SELECT 1 FROM litellm_usage.request
                                   WHERE id=$2 AND closed_at IS NULL AND instance_id=$8)
                    RETURNING *
                ), recovery AS (
                    INSERT INTO litellm_usage.recovery(attempt_id) SELECT id FROM inserted RETURNING attempt_id
                ), counted AS (
                    UPDATE litellm_usage.control SET unknown_attempts=unknown_attempts+1
                    WHERE id=1 AND EXISTS(SELECT 1 FROM recovery) RETURNING id
                )
                SELECT inserted.* FROM inserted CROSS JOIN counted
                """,
                parameters=(
                    attempt_id,
                    request.id,
                    model,
                    group,
                    deployment,
                    provider,
                    self.max_pending,
                    self.instance_id,
                ),
            ),
        )
        if not open_rows:
            raise RuntimeError("request journal closed; provider call refused")
        attempts = ATTEMPTS.validate_python(rows)
        if not attempts:
            raise RuntimeError("request journal closed or usage journal capacity exhausted; provider call refused")
        return attempts[0]

    async def finish(self, request: RequestRow, attempt: AttemptRow, body: Attempt) -> Attempt:
        async with self.transaction() as tx:
            rows = ATTEMPTS.validate_python(
                await tx.query_raw("SELECT * FROM litellm_usage.attempt WHERE id=$1 FOR UPDATE", attempt.id)
            )
            previous = rows[0]
            incoming = (
                body.model_copy(update=MappingProxyType({"endedAt": previous.body.endedAt}))
                if previous.body is not None
                and previous.body.outcome != "unknown"
                and previous.body.outcome == body.outcome
                else body
            )
            if previous.body == incoming:
                return incoming
            if previous.usage_known:
                if previous.body is not None and previous.body != incoming:
                    await tx.execute_raw("UPDATE litellm_usage.request SET conflicted=true WHERE id=$1", request.id)
                    await self.emit(tx, request, incoming, previous.version + 1)
                return previous.body if previous.body is not None else incoming
            completed = (
                previous.body.model_copy(update=MappingProxyType({"usage": incoming.usage}))
                if previous.body is not None and previous.body.outcome != "unknown"
                else incoming
            )
            version = previous.version + 1
            event = self.event(request, completed, version)
            await tx.execute_raw(
                """
                WITH updated AS (
                    UPDATE litellm_usage.attempt SET ended_at=$2::text::timestamptz,
                        version=$3,usage_known=$4,body=$5::jsonb WHERE id=$1 RETURNING request_id
                ), emitted AS (
                    INSERT INTO litellm_usage.outbox(event_id,version,request_id,payload,digest)
                    SELECT $6,$3,request_id,$7::jsonb,$8 FROM updated RETURNING id
                )
                UPDATE litellm_usage.control SET pending=pending+1,
                    unknown_attempts=unknown_attempts-CASE WHEN $4 THEN 1 ELSE 0 END
                WHERE id=1 AND EXISTS(SELECT 1 FROM emitted)
                """,
                attempt.id,
                completed.endedAt.isoformat() if completed.endedAt is not None else None,
                version,
                completed.usage.state == "known",
                completed.model_dump_json(),
                event.eventId,
                event.model_dump_json(),
                event.digest,
            )
            return completed

    async def close(self, request: RequestRow, outcome: str) -> None:
        async with self.transaction() as tx:
            rows = REQUESTS.validate_python(
                await tx.query_raw("SELECT * FROM litellm_usage.request WHERE id=$1 FOR NO KEY UPDATE", request.id)
            )
            if rows[0].closed_at is not None or rows[0].instance_id != self.instance_id:
                return
            attempts = ATTEMPTS.validate_python(
                await tx.query_raw(
                    "SELECT * FROM litellm_usage.attempt WHERE request_id=$1 ORDER BY ordinal", request.id
                )
            )
            ended = datetime.now(UTC)
            body = Closed.model_validate(
                MappingProxyType(
                    {
                        "type": "request.closed",
                        "endedAt": ended,
                        "outcome": outcome,
                        "attemptIds": tuple(a.id for a in attempts),
                    }
                )
            )
            event = self.event(request, body)
            await tx.execute_raw(
                """
                WITH closed AS (
                    UPDATE litellm_usage.request SET closed_at=$2::text::timestamptz,outcome=$3
                    WHERE id=$1 RETURNING id
                ), emitted AS (
                    INSERT INTO litellm_usage.outbox(event_id,version,request_id,payload,digest)
                    SELECT $4,1,id,$5::jsonb,$6 FROM closed RETURNING id
                )
                UPDATE litellm_usage.control SET open_requests=open_requests-1,pending=pending+1
                WHERE id=1 AND EXISTS(SELECT 1 FROM emitted)
                """,
                request.id,
                ended.isoformat(),
                outcome,
                event.eventId,
                event.model_dump_json(),
                event.digest,
            )

    async def claim(self) -> Sequence[Delivery]:
        (rows,) = await self.query_transaction(
            Statement(
                query="\n                WITH due AS (SELECT id FROM litellm_usage.outbox\n                    WHERE disposition='pending' AND next_attempt_at <= now()\n                      AND (lease_until IS NULL OR lease_until < now())\n                    ORDER BY next_attempt_at,id FOR UPDATE SKIP LOCKED LIMIT 100)\n                UPDATE litellm_usage.outbox o SET lease_owner=$1, lease_until=now()+interval '30 seconds'\n                FROM due WHERE o.id=due.id RETURNING o.id, o.payload, o.attempts\n            ",
                parameters=(self.instance_id,),
            )
        )
        return DELIVERIES.validate_python(rows)

    async def complete(self, results: Sequence[DeliveryResult]) -> None:
        if not results:
            return
        encoded = TypeAdapter(Sequence[DeliveryResult]).dump_json(results).decode()
        await self.query_transaction(
            Statement(
                query="""
                WITH acknowledgements AS (
                    SELECT * FROM jsonb_to_recordset($1::jsonb)
                    AS a(id bigint, status text, reason text, delay_ms integer)
                ), updated AS (
                    UPDATE litellm_usage.outbox o SET
                        disposition=CASE WHEN a.status='accepted' THEN 'delivered'
                                         WHEN a.status='retry' THEN 'pending' ELSE 'quarantined' END,
                        delivered_at=CASE WHEN a.status='accepted' THEN now() ELSE NULL END,
                        reason=a.reason, lease_owner=NULL, lease_until=NULL,
                        attempts=o.attempts + CASE WHEN a.status='retry' THEN 1 ELSE 0 END,
                        next_attempt_at=now()+a.delay_ms*interval '1 millisecond'
                    FROM acknowledgements a WHERE o.id=a.id AND o.lease_owner=$2 AND o.disposition='pending'
                    RETURNING o.request_id, o.disposition
                ), quarantined AS (
                    UPDATE litellm_usage.request r SET conflicted=true FROM updated u
                    WHERE r.id=u.request_id AND u.disposition='quarantined' RETURNING r.id
                )
                UPDATE litellm_usage.control SET pending=pending-(
                    SELECT count(*) FROM updated WHERE disposition='delivered') WHERE id=1
            """,
                parameters=(encoded, self.instance_id),
            )
        )
        for result in results:
            if result.status in ("rejected", "conflict"):
                logger.error("usage delivery quarantined id=%s reason=%s", result.id, result.reason)

    async def acknowledge(self, delivery: Delivery, status: str, reason: str = "") -> None:
        await self.complete((DeliveryResult(id=delivery.id, status=status, reason=reason),))

    @staticmethod
    def retry_result(delivery: Delivery) -> DeliveryResult:
        delay_ms = (100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 30000)[max(0, min(delivery.attempts, 9))]
        return DeliveryResult(id=delivery.id, status="retry", delay_ms=delay_ms)

    async def retry(self, delivery: Delivery) -> None:
        await self.complete((self.retry_result(delivery),))

    async def heartbeat(self, max_bytes: int) -> None:
        await self.query_transaction(
            Statement(
                query="\n                INSERT INTO litellm_usage.instance(id) VALUES ($1)\n                ON CONFLICT(id) DO UPDATE SET heartbeat_at=now()\n            ",
                parameters=(self.instance_id,),
            ),
            Statement(
                query="\n                UPDATE litellm_usage.control SET admitted=(\n                    pg_total_relation_size('litellm_usage.outbox') + pg_total_relation_size('litellm_usage.attempt')\n                    + pg_total_relation_size('litellm_usage.request') + pg_total_relation_size('litellm_usage.recovery') < $1) WHERE id=1\n            ",
                parameters=(max_bytes,),
            ),
            Statement(
                query="\n                UPDATE litellm_usage.outbox SET payload=NULL WHERE id IN (\n                    SELECT o.id FROM litellm_usage.outbox o JOIN litellm_usage.request r ON r.id=o.request_id\n                    WHERE o.disposition='delivered' AND o.delivered_at < now()-interval '24 hours'\n                      AND o.payload IS NOT NULL AND r.closed_at IS NOT NULL AND NOT r.conflicted\n                      AND NOT EXISTS(SELECT 1 FROM litellm_usage.attempt a WHERE a.request_id=r.id\n                                     AND (a.ended_at IS NULL OR NOT a.usage_known))\n                    ORDER BY o.delivered_at, o.id LIMIT 1000 FOR UPDATE OF o SKIP LOCKED)\n            "
            ),
        )

    async def orphaned(self) -> Sequence[RequestRow]:
        (rows,) = await self.query_transaction(
            Statement(
                query="""
                WITH due AS (
                    SELECT r.id FROM litellm_usage.request r
                    LEFT JOIN litellm_usage.instance i ON i.id=r.instance_id
                    WHERE r.closed_at IS NULL AND (
                        (r.instance_id=$1 AND r.outcome='unknown') OR (
                            r.started_at < now()-interval '60 seconds'
                            AND (i.heartbeat_at IS NULL OR i.heartbeat_at < now()-interval '60 seconds')
                        )
                    )
                    ORDER BY r.started_at LIMIT 100 FOR UPDATE OF r SKIP LOCKED
                ) UPDATE litellm_usage.request r SET instance_id=$1, outcome='unknown' FROM due
                  WHERE r.id=due.id RETURNING r.*
            """,
                parameters=(self.instance_id,),
            )
        )
        return REQUESTS.validate_python(rows)

    async def observe(self, request: RequestRow, body: Observation) -> None:
        async with self.transaction() as tx:
            rows = await tx.query_raw(
                """
                SELECT observation, observation_version FROM litellm_usage.recovery
                WHERE attempt_id=$1 FOR UPDATE
            """,
                body.attemptId,
            )
            previous = TypeAdapter(Sequence[Mapping[str, object]]).validate_python(rows)[0]
            prior = Observation.model_validate(previous["observation"]) if previous["observation"] is not None else None
            observed = _retain_native_first_response(prior, body)
            if prior is not None:
                if prior == observed:
                    await tx.execute_raw(
                        """
                        UPDATE litellm_usage.recovery SET completed=$2 AND EXISTS(
                            SELECT 1 FROM litellm_usage.attempt WHERE id=$1 AND usage_known)
                        WHERE attempt_id=$1
                    """,
                        observed.attemptId,
                        isinstance(observed.cost, KnownCost),
                    )
                    return
                if (
                    isinstance(prior.cost, KnownCost)
                    or prior.model_copy(update=MappingProxyType({"cost": observed.cost})) != observed
                ):
                    await tx.execute_raw("UPDATE litellm_usage.request SET conflicted=true WHERE id=$1", request.id)
                    version = TypeAdapter(int).validate_python(previous["observation_version"]) + 1
                    await self.emit(tx, request, observed, version)
                    await tx.execute_raw(
                        """
                        UPDATE litellm_usage.recovery SET observation_version=$2,completed=true WHERE attempt_id=$1
                    """,
                        observed.attemptId,
                        version,
                    )
                    logger.error("supplier evidence conflict request=%s attempt=%s", request.id, observed.attemptId)
                    return
            version = TypeAdapter(int).validate_python(previous["observation_version"]) + 1
            event = self.event(request, observed, version)
            await tx.execute_raw(
                """
                WITH updated AS (
                    UPDATE litellm_usage.recovery SET observation=$2::jsonb, observation_version=$3,
                        completed=$4 AND EXISTS(SELECT 1 FROM litellm_usage.attempt WHERE id=$1 AND usage_known)
                    WHERE attempt_id=$1 RETURNING attempt_id
                ), emitted AS (
                    INSERT INTO litellm_usage.outbox(event_id,version,request_id,payload,digest)
                    SELECT $5,$3,$6,$7::jsonb,$8 FROM updated RETURNING id
                )
                UPDATE litellm_usage.control SET pending=pending+1
                WHERE id=1 AND EXISTS(SELECT 1 FROM emitted)
            """,
                observed.attemptId,
                observed.model_dump_json(),
                version,
                isinstance(observed.cost, KnownCost),
                event.eventId,
                request.id,
                event.model_dump_json(),
                event.digest,
            )

    async def recovery_candidates(self) -> Sequence[AttemptRow]:
        (rows,) = await self.query_transaction(
            Statement(
                query="""
                WITH due AS (
                    SELECT c.attempt_id FROM litellm_usage.recovery c
                    JOIN litellm_usage.attempt a ON a.id=c.attempt_id
                    JOIN litellm_usage.request r ON r.id=a.request_id
                    WHERE NOT c.completed AND c.next_at <= now() AND r.closed_at IS NOT NULL
                      AND (c.lease_until IS NULL OR c.lease_until < now())
                    ORDER BY c.next_at,c.attempt_id LIMIT 100 FOR UPDATE OF c SKIP LOCKED
                ), claimed AS (
                    UPDATE litellm_usage.recovery c SET lease_owner=$1,
                        lease_until=now()+interval '30 seconds', attempts=attempts+1,
                        next_at=now()+LEAST(3600, power(2,LEAST(attempts,12))) * interval '1 second'
                    FROM due WHERE c.attempt_id=due.attempt_id RETURNING c.attempt_id
                ) SELECT a.* FROM litellm_usage.attempt a JOIN claimed c ON c.attempt_id=a.id
            """,
                parameters=(self.instance_id,),
            )
        )
        return ATTEMPTS.validate_python(rows)

    async def request_for(self, attempt: AttemptRow) -> RequestRow:
        rows = await self.database.writer_db.query_raw(
            "SELECT * FROM litellm_usage.request WHERE id=$1", attempt.request_id
        )
        return REQUESTS.validate_python(rows)[0]

    async def retained_source(self, attempt: AttemptRow) -> Mapping[str, object] | None:
        retained = await self.database.writer_db.query_raw(
            "SELECT source_payload FROM litellm_usage.recovery WHERE attempt_id=$1", attempt.id
        )
        stored = TypeAdapter(Sequence[Mapping[str, object]]).validate_python(retained)
        return (
            TypeAdapter(Mapping[str, object]).validate_python(stored[0]["source_payload"])
            if stored and stored[0]["source_payload"] is not None
            else None
        )

    async def spend_log(self, attempt: AttemptRow) -> Mapping[str, object] | None:
        source = await self.retained_source(attempt)
        tables = await self.database.writer_db.query_raw("SELECT to_regclass('\"LiteLLM_SpendLogs\"')::text AS name")
        if not TypeAdapter(Sequence[Mapping[str, str | None]]).validate_python(tables)[0]["name"]:
            return source
        rows = await self.database.writer_db.query_raw(
            """
            SELECT request_id, "startTime", "endTime", "completionStartTime", prompt_tokens,
                   completion_tokens, metadata, spend, model_id, status, call_type
            FROM "LiteLLM_SpendLogs" WHERE request_id=$1 LIMIT 1
        """,
            attempt.log_id,
        )
        found = TypeAdapter(Sequence[Mapping[str, object]]).validate_python(rows)
        if not found:
            return source
        return MappingProxyType({**found[0], "journal_stream": source.get("journal_stream") if source else None})

    async def retain_source(self, attempt: AttemptRow, payload: Mapping[str, object]) -> None:
        codec = TypeAdapter(Mapping[str, object])
        encoded = codec.dump_json(payload, fallback=codec.validate_python).decode()
        await self.database.writer_db.execute_raw(
            """
            UPDATE litellm_usage.recovery SET source_payload=$2::jsonb
            WHERE attempt_id=$1 AND source_payload IS NULL
        """,
            attempt.id,
            encoded,
        )
