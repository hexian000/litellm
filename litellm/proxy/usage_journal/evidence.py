from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from types import MappingProxyType

from pydantic import TypeAdapter, ValidationError

from .lifecycle import JOURNAL_ERRORS, OBJECTS, fields
from .protocol import Attempt, KnownCost, KnownUsage, Observation, UnknownCost, UnknownUsage
from .store import ATTEMPTS, AttemptRow, Store

logger = logging.getLogger(__name__)


def moment(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
        except ValueError:
            return None
    return None


def metadata(payload: Mapping[str, object]) -> Mapping[str, object]:
    raw = payload.get("metadata")
    if isinstance(raw, str):
        try:
            return OBJECTS.validate_json(raw)
        except ValueError:
            return MappingProxyType({})
    return fields(raw)


def cost_from(payload: Mapping[str, object]) -> KnownCost | UnknownCost:
    if fields(metadata(payload).get("additional_usage_values")).get("journal_cost_known") is False:
        return UnknownCost(state="unknown", reason="supplier_cost_unproven")
    raw = payload.get("spend")
    if raw is None or isinstance(raw, bool):
        return UnknownCost(state="unknown", reason="supplier_cost_missing")
    try:
        amount = Decimal(str(raw))
        if not amount.is_finite() or amount < 0 or amount >= Decimal("1000000000000"):
            return UnknownCost(state="unknown", reason="supplier_cost_invalid")
        if (
            amount == 0
            and fields(metadata(payload).get("additional_usage_values")).get("journal_cost_known") is not True
        ):
            return UnknownCost(state="unknown", reason="supplier_cost_unproven_zero")
        return KnownCost(
            state="known",
            amount=format(amount.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP), "f"),
            source="spend_log",
        )
    except (InvalidOperation, ValueError):
        return UnknownCost(state="unknown", reason="supplier_cost_invalid")


def status_code(payload: Mapping[str, object]) -> int | None:
    error = fields(metadata(payload).get("error_information"))
    for value in (payload.get("status_code"), error.get("status_code"), error.get("error_code")):
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and 100 <= value <= 599:
            return value
        if isinstance(value, str) and value.isdigit() and 100 <= int(value) <= 599:
            return int(value)
    return None


async def apply_source(store: Store, attempt: AttemptRow, payload: Mapping[str, object]) -> None:
    request = await store.request_for(attempt)
    if payload.get("request_id") != attempt.log_id:
        raise ValueError("source log identity does not match journal")
    deployment = payload.get("model_id")
    if deployment and attempt.deployment_id is not None and deployment != attempt.deployment_id:
        raise ValueError("source deployment does not match journal")
    body = attempt.body
    if not attempt.usage_known and request.closed_at is not None:
        meta = metadata(payload)
        try:
            reported = TypeAdapter(KnownUsage | UnknownUsage).validate_python(
                fields(meta.get("additional_usage_values")).get("journal_usage")
            )
        except ValidationError:
            reported = UnknownUsage(state="unknown", reason="supplier_usage_unproven")
        ended = moment(payload.get("endTime"))
        status = payload.get("status")
        failed = status in frozenset(("failure", "failed", "error"))
        if (isinstance(reported, KnownUsage) or failed) and ended is not None and ended >= attempt.started_at:
            recovered = Attempt(
                type="attempt.finished",
                attemptId=attempt.id,
                logId=attempt.log_id,
                ordinal=attempt.ordinal,
                kind=attempt.kind,
                model=attempt.model,
                modelGroup=attempt.model_group,
                deploymentId=attempt.deployment_id,
                provider=attempt.provider,
                startedAt=attempt.started_at,
                endedAt=ended,
                outcome="failure" if failed else "success",
                usage=reported,
                supplierAttempted=True,
            )
            body = await store.finish(request, attempt, recovered)
    if body is None or body.endedAt is None or body.outcome == "unknown":
        return
    first = moment(payload.get("completionStartTime"))
    first = first if first is not None and body.startedAt <= first <= body.endedAt else None
    call = payload.get("call_type")
    streaming = payload.get("journal_stream")
    await store.observe(
        request,
        Observation(
            type="attempt.observed",
            attemptId=attempt.id,
            logId=attempt.log_id,
            deploymentId=attempt.deployment_id,
            startedAt=body.startedAt,
            endedAt=body.endedAt,
            firstResponseAt=first,
            outcome=body.outcome,
            supplierAttempted=body.supplierAttempted is True and body.outcome != "cache_hit",
            requestType=call[:48] if isinstance(call, str) else "unknown",
            stream=streaming if isinstance(streaming, bool) else None,
            statusCode=status_code(payload),
            cost=cost_from(payload),
        ),
    )


async def capture(store: Store, payload: Mapping[str, object]) -> None:
    log_id = payload.get("request_id")
    if not isinstance(log_id, str):
        return
    rows = await store.database.writer_db.query_raw("SELECT * FROM litellm_usage.attempt WHERE log_id=$1", log_id)
    attempts = ATTEMPTS.validate_python(rows)
    if not attempts:
        return
    selected = MappingProxyType(
        {
            key: payload.get(key)
            for key in (
                "request_id",
                "endTime",
                "completionStartTime",
                "prompt_tokens",
                "completion_tokens",
                "spend",
                "model_id",
                "status",
                "call_type",
                "journal_stream",
            )
        }
    )
    meta = metadata(payload)
    source = MappingProxyType(
        {
            **selected,
            "status_code": status_code(payload),
            "metadata": MappingProxyType(
                {
                    "usage_object": meta.get("usage_object"),
                    "additional_usage_values": meta.get("additional_usage_values"),
                }
            ),
        }
    )
    await store.retain_source(attempts[0], source)
    current = (
        attempts[0]
        if attempts[0].usage_known
        else ATTEMPTS.validate_python(
            await store.database.writer_db.query_raw("SELECT * FROM litellm_usage.attempt WHERE id=$1", attempts[0].id)
        )[0]
    )
    await apply_source(store, current, source)


async def apply_retained_source(store: Store, attempt: AttemptRow, completed: Attempt) -> None:
    try:
        source = await store.retained_source(attempt)
        if source is not None:
            current = attempt.model_copy(
                update=MappingProxyType({"body": completed, "usage_known": isinstance(completed.usage, KnownUsage)})
            )
            await apply_source(store, current, source)
    except JOURNAL_ERRORS as error:
        logger.error("supplier retained evidence pending attempt=%s error=%s", attempt.id, type(error).__name__)


async def recover_evidence(store: Store) -> int:
    attempts = await store.recovery_candidates()
    for attempt in attempts:
        try:
            payload = await store.spend_log(attempt)
            if payload is not None:
                await apply_source(store, attempt, payload)
            if payload is None or not attempt.usage_known:
                logger.warning("usage evidence recovery pending request=%s attempt=%s", attempt.request_id, attempt.id)
        except JOURNAL_ERRORS as error:
            logger.error("usage evidence recovery deferred attempt=%s error=%s", attempt.id, type(error).__name__)
    return len(attempts)
