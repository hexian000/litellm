from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

from prisma import Prisma
from pydantic import BaseModel, TypeAdapter


class MigrationReport(BaseModel):
    target: Mapping[str, object]
    version: int = 1
    changed: bool | None = None
    preview: str | None = None
    operationId: str | None = None


class Options(BaseModel):
    database: str
    apply: bool


TARGET = TypeAdapter(Sequence[Mapping[str, object]])


async def migrate(expected_database: str, apply: bool) -> None:
    db = Prisma()
    await db.connect()
    try:
        target = TARGET.validate_python(
            await db.query_raw(
                "SELECT current_database() AS database, current_user AS role, inet_server_addr()::text AS address"
            )
        )[0]
        if target["database"] != expected_database:
            raise RuntimeError("migration database target does not match --database")
        exists = await db.query_raw("SELECT to_regclass('litellm_usage.revision')::text AS revision")
        if TARGET.validate_python(exists)[0]["revision"] is not None:
            versions = await db.query_raw("SELECT version FROM litellm_usage.revision")
            if tuple(row["version"] for row in TARGET.validate_python(versions)) != (1,):
                raise RuntimeError("unexpected usage schema version; do not change this database")
            sys.stdout.write(MigrationReport(target=target, changed=False).model_dump_json(exclude_none=True))
            return
        schema = await db.query_raw(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name='litellm_usage'"
        )
        if schema:
            raise RuntimeError("unexpected existing usage schema without revision; stop and inspect")
        if not apply:
            sys.stdout.write(
                MigrationReport(target=target, preview="create independent litellm_usage schema").model_dump_json(
                    exclude_none=True
                )
            )
            return
        operation = str(uuid4())
        async with db.tx() as tx:
            await tx.execute_raw("SET LOCAL lock_timeout='5s'")
            await tx.query_raw("SELECT pg_advisory_xact_lock(117261, 1)::text")
            for statement in (Path(__file__).parent / "migrations/001.sql").read_text().split(";"):
                if statement.strip():
                    await tx.execute_raw(statement)
            await tx.execute_raw("INSERT INTO litellm_usage.revision(version, operation_id) VALUES(1, $1)", operation)
        tables = await db.query_raw(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='litellm_usage'"
        )
        if len(TARGET.validate_python(tables)) != 6:
            raise RuntimeError("usage migration postcondition failed")
        sys.stdout.write(
            MigrationReport(target=target, operationId=operation, changed=True).model_dump_json(exclude_none=True)
        )
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
