from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Protocol

from pydantic import BaseModel, TypeAdapter

from .protocol import PROTOCOL, Attempt, Closed, Started, UsageEvent

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
    kind: str
    started_at: datetime
    version: int = 0
    usage_known: bool = False
    body: Attempt | None = None


class Delivery(BaseModel):
    id: int
    payload: UsageEvent
    attempts: int


REQUESTS = TypeAdapter(Sequence[RequestRow])
ATTEMPTS = TypeAdapter(Sequence[AttemptRow])
DELIVERIES = TypeAdapter(Sequence[Delivery])


class DeliveryResult(BaseModel):
    id: int
    status: str
    reason: str = ""
    delay_ms: int = 100


class Store:
    def __init__(self, database: Database, instance_id: str, max_pending: int = 100000) -> None:
        self.database = database
        self.instance_id = instance_id
        self.max_pending = max_pending

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Sql]:
        async with self.database.tx() as tx:
            await tx.execute_raw("SET LOCAL lock_timeout = '2s'")
            await tx.execute_raw("SET LOCAL statement_timeout = '5s'")
            yield tx

    async def check_version(self) -> bool:
        rows = await self.database.writer_db.query_raw("SELECT version FROM litellm_usage.revision")
        return tuple(row["version"] for row in TypeAdapter(Sequence[Mapping[str, int]]).validate_python(rows)) == (1,)

    async def emit(self, tx: Sql, request: RequestRow, body: Started | Attempt | Closed, version: int = 1) -> None:
        suffix = f"attempt/{body.attemptId}" if isinstance(body, Attempt) else body.type.removeprefix("request.")
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
        event = parsed.model_copy(update=MappingProxyType({"digest": parsed.content_digest()}))
        await tx.execute_raw(
            "\n            INSERT INTO litellm_usage.outbox(event_id, version, request_id, payload, digest)\n            VALUES ($1, $2, $3, $4::jsonb, $5)\n        ",
            event.eventId,
            version,
            request.id,
            event.model_dump_json(),
            event.digest,
        )
        await tx.execute_raw("UPDATE litellm_usage.control SET pending = pending + 1 WHERE id = 1")

    async def start(self, admission: Admission) -> RequestRow:
        async with self.transaction() as tx:
            capacity = await tx.query_raw(
                "\n                SELECT id FROM litellm_usage.control WHERE id = 1 AND admitted AND pending + open_requests + unknown_attempts < $1 FOR UPDATE\n            ",
                self.max_pending,
            )
            if not capacity:
                raise RuntimeError("usage journal capacity exhausted")
            rows = await tx.query_raw(
                "\n                INSERT INTO litellm_usage.request(id, trace_id, project_id, customer_model, mode, instance_id)\n                VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING RETURNING *\n            ",
                admission.request_id,
                admission.trace_id,
                admission.project_id,
                admission.customer_model,
                admission.mode,
                self.instance_id,
            )
            requests = REQUESTS.validate_python(rows)
            if not requests:
                raise RuntimeError("request journal already exists; provider replay refused")
            request = requests[0]
            await tx.execute_raw("UPDATE litellm_usage.control SET open_requests=open_requests+1 WHERE id=1")
            await self.emit(tx, request, Started(type="request.started", startedAt=request.started_at))
            return request

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
        async with self.transaction() as tx:
            open_rows = await tx.query_raw(
                "SELECT id FROM litellm_usage.request WHERE id=$1 AND closed_at IS NULL FOR UPDATE", request.id
            )
            if not open_rows:
                raise RuntimeError("request journal closed; provider call refused")
            capacity = await tx.query_raw(
                "SELECT id FROM litellm_usage.control WHERE id=1 AND admitted AND pending + open_requests + unknown_attempts < $1",
                self.max_pending,
            )
            if not capacity:
                raise RuntimeError("usage journal capacity exhausted")
            rows = await tx.query_raw(
                "\n                INSERT INTO litellm_usage.attempt(id, request_id, ordinal, log_id, model, model_group,\n                                                  deployment_id, provider, kind)\n                SELECT $1, $2, count(*)::integer + 1, $1, $3, $4, $5, $6,\n                       CASE WHEN count(*) = 0 THEN 'primary'\n                            WHEN bool_and(model = $3) THEN 'retry' ELSE 'fallback' END\n                FROM litellm_usage.attempt WHERE request_id=$2 RETURNING *\n            ",
                attempt_id,
                request.id,
                model,
                group,
                deployment,
                provider,
            )
            await tx.execute_raw("UPDATE litellm_usage.control SET unknown_attempts=unknown_attempts+1 WHERE id=1")
            return ATTEMPTS.validate_python(rows)[0]

    async def finish(self, request: RequestRow, attempt: AttemptRow, body: Attempt) -> None:
        async with self.transaction() as tx:
            rows = ATTEMPTS.validate_python(
                await tx.query_raw("SELECT * FROM litellm_usage.attempt WHERE id=$1 FOR UPDATE", attempt.id)
            )
            previous = rows[0]
            if previous.usage_known:
                if previous.body is not None and previous.body != body:
                    await tx.execute_raw("UPDATE litellm_usage.request SET conflicted=true WHERE id=$1", request.id)
                    await self.emit(tx, request, body, previous.version + 1)
                return
            if previous.body is not None:
                body = previous.body.model_copy(update=MappingProxyType({"usage": body.usage}))
            version = previous.version + 1
            await tx.execute_raw(
                "\n                UPDATE litellm_usage.attempt SET ended_at=$2::timestamptz, version=$3, usage_known=$4, body=$5::jsonb\n                WHERE id=$1\n            ",
                attempt.id,
                body.endedAt.isoformat(),
                version,
                body.usage.state == "known",
                body.model_dump_json(),
            )
            if body.usage.state == "known":
                await tx.execute_raw("UPDATE litellm_usage.control SET unknown_attempts=unknown_attempts-1 WHERE id=1")
            await self.emit(tx, request, body, version)

    async def close(self, request: RequestRow, outcome: str) -> None:
        async with self.transaction() as tx:
            rows = REQUESTS.validate_python(
                await tx.query_raw("SELECT * FROM litellm_usage.request WHERE id=$1 FOR UPDATE", request.id)
            )
            if rows[0].closed_at is not None:
                return
            attempts = ATTEMPTS.validate_python(
                await tx.query_raw(
                    "SELECT * FROM litellm_usage.attempt WHERE request_id=$1 ORDER BY ordinal", request.id
                )
            )
            ended = datetime.now(UTC)
            await tx.execute_raw(
                "UPDATE litellm_usage.request SET closed_at=$2::timestamptz, outcome=$3 WHERE id=$1",
                request.id,
                ended.isoformat(),
                outcome,
            )
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
            await tx.execute_raw("UPDATE litellm_usage.control SET open_requests=open_requests-1 WHERE id=1")
            await self.emit(tx, request, body)

    async def claim(self) -> Sequence[Delivery]:
        async with self.transaction() as tx:
            rows = await tx.query_raw(
                "\n                WITH due AS (SELECT id FROM litellm_usage.outbox\n                    WHERE disposition='pending' AND next_attempt_at <= now()\n                      AND (lease_until IS NULL OR lease_until < now())\n                    ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 100)\n                UPDATE litellm_usage.outbox o SET lease_owner=$1, lease_until=now()+interval '30 seconds'\n                FROM due WHERE o.id=due.id RETURNING o.id, o.payload, o.attempts\n            ",
                self.instance_id,
            )
            return DELIVERIES.validate_python(rows)

    async def complete(self, results: Sequence[DeliveryResult]) -> None:
        if not results:
            return
        encoded = TypeAdapter(Sequence[DeliveryResult]).dump_json(results).decode()
        async with self.transaction() as tx:
            await tx.execute_raw(
                """
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
                encoded,
                self.instance_id,
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
        async with self.transaction() as tx:
            await tx.execute_raw(
                "\n                INSERT INTO litellm_usage.instance(id) VALUES ($1)\n                ON CONFLICT(id) DO UPDATE SET heartbeat_at=now()\n            ",
                self.instance_id,
            )
            await tx.execute_raw(
                "\n                UPDATE litellm_usage.control SET admitted=(\n                    pg_total_relation_size('litellm_usage.outbox') + pg_total_relation_size('litellm_usage.attempt')\n                    + pg_total_relation_size('litellm_usage.request') < $1) WHERE id=1\n            ",
                max_bytes,
            )
            await tx.execute_raw(
                "\n                UPDATE litellm_usage.outbox SET payload=NULL WHERE id IN (\n                    SELECT o.id FROM litellm_usage.outbox o JOIN litellm_usage.request r ON r.id=o.request_id\n                    WHERE o.disposition='delivered' AND o.delivered_at < now()-interval '24 hours'\n                      AND o.payload IS NOT NULL AND r.closed_at IS NOT NULL AND NOT r.conflicted\n                      AND NOT EXISTS(SELECT 1 FROM litellm_usage.attempt a WHERE a.request_id=r.id\n                                     AND (a.ended_at IS NULL OR NOT a.usage_known))\n                    ORDER BY o.delivered_at, o.id LIMIT 1000 FOR UPDATE OF o SKIP LOCKED)\n            "
            )

    async def orphaned(self) -> Sequence[RequestRow]:
        rows = await self.database.writer_db.query_raw(
            "\n            SELECT r.* FROM litellm_usage.request r LEFT JOIN litellm_usage.instance i ON i.id=r.instance_id\n            WHERE r.closed_at IS NULL AND (i.heartbeat_at IS NULL OR i.heartbeat_at < now()-interval '60 seconds')\n              AND r.started_at < now()-interval '60 seconds' ORDER BY r.started_at LIMIT 100\n        "
        )
        return REQUESTS.validate_python(rows)
