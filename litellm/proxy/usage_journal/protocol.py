"""Version-one durable usage wire contract, shared with the gateway fork."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
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
    endedAt: AwareDatetime
    outcome: Literal["success", "failure", "cancelled", "cache_hit"]
    usage: Annotated[KnownUsage | UnknownUsage, Field(discriminator="state")]

    @model_validator(mode="after")
    def time_order(self) -> Attempt:
        if self.endedAt < self.startedAt:
            raise ValueError("attempt ends before it starts")
        return self


class Closed(WireModel):
    type: Literal["request.closed"]
    endedAt: AwareDatetime
    outcome: Literal["success", "failure", "cancelled"]
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
    body: Annotated[Started | Attempt | Closed, Field(discriminator="type")]
    digest: Annotated[str, Field(pattern="^[a-f0-9]{64}$")]

    def content_digest(self) -> str:
        payload = self.model_dump(mode="json", exclude=MappingProxyType({"digest": True}))
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()


class UsageBatch(WireModel):
    events: Annotated[Sequence[UsageEvent], Field(min_length=1, max_length=100)]
