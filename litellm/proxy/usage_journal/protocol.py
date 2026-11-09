"""Version-one durable usage wire contract, shared with the gateway fork."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from functools import cache
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

PROTOCOL = "litellm-usage/1"
Handle = Annotated[str, Field(min_length=1, max_length=191, pattern="^[A-Za-z0-9_.:/-]+$")]
Count = Annotated[int, Field(strict=True, ge=0, le=2**53 - 1)]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KnownUsage(WireModel):
    state: Literal["known"]
    promptTokens: Count
    completionTokens: Count
    cacheReadTokens: Count

    @model_validator(mode="after")
    def cache_subset(self) -> KnownUsage:
        if self.cacheReadTokens > self.promptTokens:
            raise ValueError("cache reads exceed prompt tokens")
        return self


class UnknownUsage(WireModel):
    state: Literal["unknown"]
    reason: Annotated[str, Field(min_length=1, max_length=128)]


class KnownCost(WireModel):
    state: Literal["known"]
    amount: Annotated[str, Field(pattern=r"^(?:0|[1-9][0-9]{0,11})(?:\.[0-9]{1,6})?$")]
    currency: Literal["USD"] = "USD"
    source: Literal["spend_log", "logging_payload"]


class UnknownCost(WireModel):
    state: Literal["unknown"]
    reason: Annotated[str, Field(min_length=1, max_length=128)]


class Observation(WireModel):
    type: Literal["attempt.observed"]
    attemptId: Handle
    logId: Annotated[str, Field(min_length=1, max_length=512)]
    deploymentId: Handle | None
    providerModelId: Handle | None = None
    startedAt: AwareDatetime
    endedAt: AwareDatetime
    firstResponseAt: AwareDatetime | None = None
    outcome: Literal["success", "failure", "cancelled", "cache_hit"]
    statusCode: Annotated[int, Field(strict=True, ge=100, le=599)] | None = None
    supplierAttempted: bool
    requestType: Annotated[str, Field(min_length=1, max_length=48)]
    stream: bool | None
    cost: Annotated[KnownCost | UnknownCost, Field(discriminator="state")]

    @model_validator(mode="after")
    def ordered_times(self) -> Observation:
        if self.endedAt < self.startedAt or (
            self.firstResponseAt is not None and not self.startedAt <= self.firstResponseAt <= self.endedAt
        ):
            raise ValueError("observation times are not ordered")
        return self


class Started(WireModel):
    type: Literal["request.started"]
    startedAt: AwareDatetime


class Attempt(WireModel):
    type: Literal["attempt.finished"]
    attemptId: Handle
    logId: Annotated[str, Field(min_length=1, max_length=512)]
    ordinal: Annotated[int, Field(strict=True, ge=1, le=10000)]
    kind: Literal["primary", "retry", "fallback"]
    model: Handle
    modelGroup: Handle
    deploymentId: Handle | None
    provider: Annotated[str, Field(max_length=64)] | None
    startedAt: AwareDatetime
    endedAt: AwareDatetime | None
    outcome: Literal["success", "failure", "cancelled", "cache_hit", "unknown"]
    usage: Annotated[KnownUsage | UnknownUsage, Field(discriminator="state")]
    supplierAttempted: bool | None = None

    @model_validator(mode="after")
    def time_order(self) -> Attempt:
        if self.endedAt is not None and self.endedAt < self.startedAt:
            raise ValueError("attempt ends before it starts")
        if (self.outcome == "unknown") != (self.endedAt is None):
            raise ValueError("unknown completion must not invent an end timestamp")
        return self


class Closed(WireModel):
    type: Literal["request.closed"]
    endedAt: AwareDatetime
    outcome: Literal["success", "failure", "cancelled", "unknown"]
    attemptIds: Annotated[Sequence[Handle], Field(max_length=10000)]

    @model_validator(mode="after")
    def unique_attempts(self) -> Closed:
        if len(frozenset(self.attemptIds)) != len(self.attemptIds):
            raise ValueError("duplicate attempt identity")
        return self


class UsageEvent(WireModel):
    protocol: Literal["litellm-usage/1"]
    eventId: Handle
    version: Annotated[int, Field(strict=True, ge=1, le=2147483647)]
    occurredAt: AwareDatetime
    instanceId: Handle
    requestId: Annotated[str, Field(min_length=26, max_length=26)]
    traceId: Handle
    projectId: Annotated[str, Field(min_length=1, max_length=191)]
    customerModel: Annotated[str, Field(min_length=1, max_length=128)]
    mode: Literal["shadow", "active"]
    body: Annotated[Started | Attempt | Closed | Observation, Field(discriminator="type")]
    digest: Annotated[str, Field(pattern="^[a-f0-9]{64}$")]

    def content_digest(self) -> str:
        payload = self.model_dump(mode="json", exclude=MappingProxyType({"digest": True}))
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()


class UsageBatch(WireModel):
    events: Annotated[Sequence[UsageEvent], Field(min_length=1, max_length=100)]


@cache
def contract_digest() -> str:
    return hashlib.sha256(
        json.dumps(
            UsageBatch.model_json_schema(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
