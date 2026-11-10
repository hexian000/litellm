from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, BeforeValidator, Field, InstanceOf, TypeAdapter

Scalar = str | int | bool | None
Rows = tuple[Mapping[str, object], ...]
ROWS = TypeAdapter(Rows)


class Statement(BaseModel, frozen=True):
    query: str
    parameters: tuple[Scalar, ...] = ()


TIMEOUTS = Statement(query="SELECT set_config('lock_timeout','2s',true),set_config('statement_timeout','5s',true)")


class Atomic(Protocol):
    async def __call__(self, statements: tuple[Statement, ...]) -> tuple[Rows, ...]: ...


@runtime_checkable
class Engine(Protocol):
    async def query(self, content: str, *, tx_id: str | None) -> object: ...


class EngineAccess(BaseModel, arbitrary_types_allowed=True, from_attributes=True):
    engine: InstanceOf[Engine] = Field(validation_alias="_engine")


class _Query(BaseModel):
    query: str
    variables: Mapping[str, object]


class _Envelope(BaseModel):
    batch: tuple[_Query, ...]
    transaction: Literal[True] = True


class _Data(BaseModel):
    result: Sequence[Mapping[str, object]]


class _Reply(BaseModel):
    data: _Data | None = None
    errors: tuple[Mapping[str, object], ...] = ()


class _Response(BaseModel):
    batchResult: tuple[_Reply, ...]


def _encode(statement: Statement) -> _Query:
    from prisma.builder import QueryBuilder

    return _Query(
        query=QueryBuilder(method="query_raw", model=None, arguments=statement.model_dump()).build_query(),
        variables=MappingProxyType({}),
    )


@cache
def _decoder() -> TypeAdapter[Rows]:
    from prisma._raw_query import deserialize_raw_results

    return TypeAdapter(Annotated[Rows, BeforeValidator(deserialize_raw_results)])


def _decode(reply: _Reply) -> Rows:
    if reply.errors or reply.data is None:
        raise RuntimeError("Prisma atomic SQL statement failed")
    return _decoder().validate_python(reply.data.result)


@dataclass(frozen=True, slots=True)
class PrismaBatch:
    engine: Callable[[], Engine]

    async def __call__(self, statements: tuple[Statement, ...]) -> tuple[Rows, ...]:
        envelope = _Envelope(batch=tuple(_encode(statement) for statement in (TIMEOUTS, *statements)))
        response = _Response.model_validate(await self.engine().query(envelope.model_dump_json(), tx_id=None))
        if len(response.batchResult) != len(statements) + 1:
            raise RuntimeError("Prisma atomic SQL returned an incomplete result")
        return tuple(_decode(reply) for reply in response.batchResult)[1:]
