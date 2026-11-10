import asyncio
import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma import Prisma
from prisma.errors import PrismaError
from prisma.types import DatasourceOverride

from litellm.proxy.db.prisma_batch import EngineAccess, PrismaBatch, Statement
from litellm.proxy.db.prisma_client import default_prisma_http_config


@pytest_asyncio.fixture(loop_scope="function")
async def client() -> AsyncIterator[Prisma]:
    url = os.environ.get("LITELLM_USAGE_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires isolated PostgreSQL LITELLM_USAGE_TEST_DATABASE_URL")
    database = Prisma(datasource=DatasourceOverride(url=url), http=default_prisma_http_config())
    await database.connect()
    try:
        yield database
    finally:
        await database.disconnect()


@pytest.fixture
def batch(client: Prisma) -> PrismaBatch:
    return PrismaBatch(lambda: EngineAccess.model_validate(client).engine)


async def test_batch_executes_in_order_and_rolls_back_as_one_transaction(client: Prisma, batch: PrismaBatch) -> None:
    name = f"batch_test_{uuid4().hex}"
    await client.execute_raw(f'CREATE TABLE "{name}" (id integer PRIMARY KEY)')
    try:
        result = await batch(
            (
                Statement(query=f'INSERT INTO "{name}" VALUES ($1) RETURNING id', parameters=(1,)),
                Statement(query=f'SELECT id FROM "{name}"'),
            )
        )
        assert result == (({"id": 1},), ({"id": 1},))
        assert await batch((Statement(query=f'UPDATE "{name}" SET id=1 WHERE false'),)) == ((),)
        with pytest.raises((PrismaError, RuntimeError)):
            await batch(
                (
                    Statement(query=f'INSERT INTO "{name}" VALUES ($1) RETURNING id', parameters=(2,)),
                    Statement(query="SELECT 1/0"),
                )
            )
        assert await client.query_raw(f'SELECT id FROM "{name}"') == [{"id": 1}]
    finally:
        await client.execute_raw(f'DROP TABLE "{name}"')


async def test_batch_preserves_parameter_values(batch: PrismaBatch) -> None:
    text = "quotes '\" \\\n汉字"
    result = await batch(
        (
            Statement(
                query="SELECT $1::text AS text,$2::bigint AS number,$3::boolean AS flag,$4::text AS empty",
                parameters=(text, 2**45, True, None),
            ),
        )
    )
    assert result == (({"text": text, "number": 2**45, "flag": True, "empty": None},),)


async def test_batch_timeouts_are_local_to_its_transaction(client: Prisma, batch: PrismaBatch) -> None:
    query = "SELECT current_setting('lock_timeout') AS lock_timeout,current_setting('statement_timeout') AS statement_timeout"
    original = await client.query_raw(query)
    result = await batch((Statement(query=query),))
    assert result == (({"lock_timeout": "2s", "statement_timeout": "5s"},),)
    assert await client.query_raw(query) == original


async def test_batch_statement_deadline_rolls_back_and_leaves_client_usable(client: Prisma, batch: PrismaBatch) -> None:
    with pytest.raises((PrismaError, RuntimeError)):
        await batch(
            (
                Statement(query="SELECT set_config('statement_timeout','40ms',true)"),
                Statement(query="SELECT pg_sleep(0.5)"),
            )
        )
    assert await client.query_raw("SELECT 1 AS value") == [{"value": 1}]


async def test_batch_resolves_the_current_engine_after_reconnect(client: Prisma, batch: PrismaBatch) -> None:
    assert await batch((Statement(query="SELECT 1 AS value"),)) == (({"value": 1},),)
    await client.disconnect()
    await client.connect()
    assert await batch((Statement(query="SELECT 2 AS value"),)) == (({"value": 2},),)


async def test_cancelled_batch_does_not_close_the_client(client: Prisma, batch: PrismaBatch) -> None:
    pending = asyncio.create_task(batch((Statement(query="SELECT pg_sleep(0.2)"),)))
    await asyncio.sleep(0.02)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert await client.query_raw("SELECT 1 AS value") == [{"value": 1}]
