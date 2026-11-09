from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from prisma import Prisma
from pydantic import BaseModel, TypeAdapter

from .store import Sql


class MigrationReport(BaseModel):
    target: Mapping[str, object]
    version: int = 2
    changed: bool | None = None
    preview: str | None = None
    operationId: str | None = None


class Options(BaseModel):
    database: str
    apply: bool


TARGET = TypeAdapter(Sequence[Mapping[str, object]])
BASE_TABLES = frozenset({"revision", "instance", "request", "attempt", "outbox", "control"})
FEATURE_TABLES = BASE_TABLES | frozenset(("feature_revision", "recovery"))


async def current_version(tx: Sql) -> int:
    exists = TARGET.validate_python(
        await tx.query_raw("SELECT to_regclass('litellm_usage.revision')::text AS revision")
    )[0]["revision"]
    if exists is None:
        if await tx.query_raw("SELECT schema_name FROM information_schema.schemata WHERE schema_name='litellm_usage'"):
            raise RuntimeError("unexpected existing usage schema without revision")
        return 0
    versions = TARGET.validate_python(await tx.query_raw("SELECT version FROM litellm_usage.revision"))
    if tuple(row["version"] for row in versions) != (1,):
        raise RuntimeError("unexpected usage schema version")
    feature = TARGET.validate_python(
        await tx.query_raw("SELECT to_regclass('litellm_usage.feature_revision')::text AS revision")
    )[0]["revision"]
    if feature is None:
        return 1
    features = TARGET.validate_python(await tx.query_raw("SELECT version FROM litellm_usage.feature_revision"))
    if tuple(row["version"] for row in features) != (2,):
        raise RuntimeError("unexpected journal feature revision")
    return 2


async def validate_tables(tx: Sql, version: int) -> None:
    if not version:
        return
    tables = TARGET.validate_python(
        await tx.query_raw("SELECT table_name FROM information_schema.tables WHERE table_schema='litellm_usage'")
    )
    if frozenset(row["table_name"] for row in tables) != (FEATURE_TABLES if version == 2 else BASE_TABLES):
        raise RuntimeError("unexpected existing journal tables")
    if version != 2:
        return
    columns = TARGET.validate_python(
        await tx.query_raw("""
        SELECT column_name,data_type FROM information_schema.columns
        WHERE table_schema='litellm_usage' AND table_name='recovery'
    """)
    )
    expected = MappingProxyType(
        {
            "attempt_id": "text",
            "next_at": "timestamp with time zone",
            "lease_owner": "text",
            "lease_until": "timestamp with time zone",
            "attempts": "integer",
            "completed": "boolean",
            "observation_version": "integer",
            "observation": "jsonb",
            "source_payload": "jsonb",
        }
    )
    if MappingProxyType({row["column_name"]: row["data_type"] for row in columns}) != expected:
        raise RuntimeError("unexpected recovery column definitions")
    index = TARGET.validate_python(
        await tx.query_raw("""
        SELECT i.indisvalid,pg_get_indexdef(i.indexrelid) AS definition FROM pg_index i
        WHERE i.indexrelid=to_regclass('litellm_usage.usage_recovery_due')
    """)
    )
    definition = (
        "CREATE INDEX usage_recovery_due ON litellm_usage.recovery USING btree "
        "(next_at, attempt_id) WHERE (NOT completed)"
    )
    if not index or index[0]["indisvalid"] is not True or index[0]["definition"] != definition:
        raise RuntimeError("unexpected or invalid recovery index")


async def apply_steps(tx: Sql, current: int, operation: str) -> None:
    if current == 1:
        existing = TARGET.validate_python(await tx.query_raw("SELECT id FROM litellm_usage.attempt LIMIT 1001"))
        if len(existing) > 1000:
            raise RuntimeError("journal upgrade exceeds 1000-attempt seed bound; inspect before migration")
    for version in range(current + 1, 3):
        script = Path(__file__).parent / f"migrations/{version:03}.sql"
        for statement in script.read_text().split(";"):
            if statement.strip():
                await tx.execute_raw(statement)
        table = "revision" if version == 1 else "feature_revision"
        await tx.execute_raw(
            f"INSERT INTO litellm_usage.{table}(version, operation_id) VALUES($1,$2)", version, operation
        )
    if current == 1:
        await tx.execute_raw("INSERT INTO litellm_usage.recovery(attempt_id) SELECT id FROM litellm_usage.attempt")
    await validate_tables(tx, 2)


async def migrate(expected_database: str, apply: bool) -> None:
    db = Prisma()
    await db.connect()
    try:
        target = TARGET.validate_python(
            await db.query_raw(
                "SELECT current_database() AS database,current_user AS role,inet_server_addr()::text AS address"
            )
        )[0]
        if target["database"] != expected_database:
            raise RuntimeError("migration database target does not match --database")
        operation = str(uuid4())
        async with db.tx() as tx:
            await tx.execute_raw("SET LOCAL lock_timeout='5s'")
            await tx.execute_raw("SET LOCAL statement_timeout='5s'")
            await tx.query_raw("SELECT pg_advisory_xact_lock(117261, 1)::text")
            current = await current_version(tx)
            await validate_tables(tx, current)
            if current == 2:
                report = MigrationReport(target=target, changed=False)
            elif not apply:
                report = MigrationReport(
                    target=target, preview=f"upgrade independent usage journal from {current} to 2"
                )
            else:
                await apply_steps(tx, current, operation)
                report = MigrationReport(target=target, changed=True, operationId=operation)
        sys.stdout.write(report.model_dump_json(exclude_none=True))
    finally:
        await db.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--apply", action="store_true")
    args = Options.model_validate(vars(parser.parse_args()))
    asyncio.run(migrate(args.database, args.apply))


if __name__ == "__main__":
    main()
