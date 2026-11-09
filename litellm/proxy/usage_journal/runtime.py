from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from uuid import uuid4

import httpx
from fastapi import HTTPException, Request
from prisma.errors import PrismaError
from pydantic import BaseModel, TypeAdapter

from .lifecycle import AttemptHandle, RequestHandle, configure, persist
from .protocol import PROTOCOL, UnknownUsage, UsageBatch, contract_digest
from .runtime_types import IncomingRequest
from .store import ATTEMPTS, Database, Delivery, DeliveryResult, Store

logger = logging.getLogger(__name__)


class Acknowledgement(BaseModel):
    eventId: str
    version: int
    digest: str
    status: str
    reason: str = ""


class Acknowledgements(BaseModel):
    acknowledgements: Sequence[Acknowledgement]


JSON = TypeAdapter(Mapping[str, object])


def authorize(request: IncomingRequest) -> None:
    token = os.environ.get("LITELLM_USAGE_REPORT_TOKEN", "")
    supplied = request.headers.get("x-guangrouter-usage-token", "")
    if not token or not secrets.compare_digest(token, supplied):
        raise HTTPException(403, "Invalid usage journal service credential")
    if (
        request.headers.get("x-guangrouter-usage-protocol") != PROTOCOL
        or request.headers.get("x-guangrouter-usage-contract") != contract_digest()
    ):
        raise HTTPException(409, "Unsupported usage protocol")


async def deliver(store: Store, client: httpx.AsyncClient, url: str, token: str) -> int:
    deliveries = await store.claim()
    if not deliveries:
        return 0
    try:
        response = await client.post(
            url,
            json=UsageBatch(events=tuple(d.payload for d in deliveries)).model_dump(mode="json"),
            headers=MappingProxyType(
                {"Authorization": f"Bearer {token}", "X-GuangRouter-Usage-Contract": contract_digest()}
            ),
        )
        if response.status_code in frozenset((400, 422)):
            await store.complete(
                tuple(
                    DeliveryResult(id=d.id, status="rejected", reason="receiver_protocol_rejection") for d in deliveries
                )
            )
            return 0
        response.raise_for_status()
        acknowledgements = Acknowledgements.model_validate(response.json())
    except (httpx.HTTPError, ValueError) as error:
        logger.error("usage HTTP delivery pending error=%s", type(error).__name__)
        await store.complete(tuple(store.retry_result(delivery) for delivery in deliveries))
        return 0
    by_key = MappingProxyType({(a.eventId, a.version, a.digest): a for a in acknowledgements.acknowledgements})

    def result_for(delivery: Delivery) -> DeliveryResult:
        event = delivery.payload
        ack = by_key.get((event.eventId, event.version, event.digest))
        if ack is None or ack.status not in ("accepted", "conflict", "rejected"):
            return store.retry_result(delivery)
        return DeliveryResult(id=delivery.id, status=ack.status, reason=ack.reason)

    await store.complete(tuple(result_for(delivery) for delivery in deliveries))
    return len(deliveries)


async def recover(store: Store) -> None:
    for row in await store.orphaned():
        handle = RequestHandle(store, row)
        attempts = ATTEMPTS.validate_python(
            await store.database.writer_db.query_raw(
                "SELECT * FROM litellm_usage.attempt WHERE request_id=$1 AND ended_at IS NULL", row.id
            )
        )
        for attempt in attempts:
            await persist(
                handle,
                AttemptHandle(
                    attempt, usage=UnknownUsage(state="unknown", reason="process_lost"), outcome="unknown", called=None
                ),
            )
        await store.close(row, "unknown")


async def run(store: Store, url: str, token: str) -> None:
    async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
        last_maintenance = datetime.min.replace(tzinfo=UTC)
        delay = 0.05
        while True:
            try:
                now = datetime.now(UTC)
                if (now - last_maintenance).total_seconds() >= 10:
                    await store.heartbeat(int(os.getenv("LITELLM_USAGE_MAX_BYTES", str(10 * 1024**3))))
                    await recover(store)
                    from .evidence import recover_evidence

                    await recover_evidence(store)
                    last_maintenance = now
                processed = await deliver(store, client, url, token)
                delay = 0.05 if processed else min(1.0, delay * 2)
            except (PrismaError, httpx.HTTPError, OSError, RuntimeError, ValueError) as error:
                logger.error("usage worker retrying error=%s", type(error).__name__)
                await asyncio.sleep(1)
            await asyncio.sleep(delay)


@dataclass
class RuntimeState:
    store: Store | None = None
    task: asyncio.Task[None] | None = None


_state = RuntimeState()


async def start(database: Database | None) -> None:
    url = os.environ.get("LITELLM_USAGE_REPORT_URL", "")
    token = os.environ.get("LITELLM_USAGE_REPORT_TOKEN", "")
    if not url:
        return
    if database is None or not token:
        raise RuntimeError("usage delivery requires the existing database and service token")
    store = Store(database, f"{socket.gethostname()}-{uuid4()}", int(os.getenv("LITELLM_USAGE_MAX_PENDING", "100000")))
    if not await store.check_version():
        raise RuntimeError("usage journal migrations 001 and 002 are required before candidate startup")
    await store.heartbeat(int(os.getenv("LITELLM_USAGE_MAX_BYTES", str(10 * 1024**3))))
    _state.store = store
    configure(store)
    _state.task = asyncio.create_task(run(store, url, token), name="usage_outbox")


async def stop() -> None:
    if _state.task is not None:
        _state.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _state.task
    configure(None)
    _state.task, _state.store = (None, None)


async def capability(request: Request) -> Mapping[str, str]:
    token = os.environ.get("LITELLM_USAGE_REPORT_TOKEN", "")
    supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not token or not secrets.compare_digest(token, supplied):
        raise HTTPException(403, "Invalid usage service credential")
    if _state.store is None:
        raise HTTPException(503, "Durable usage journal is unavailable")
    return MappingProxyType(
        {"protocol": PROTOCOL, "instanceId": _state.store.instance_id, "contractDigest": contract_digest()}
    )
