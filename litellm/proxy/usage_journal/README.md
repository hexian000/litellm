# Durable usage protocol v1

This fork journals admitted requests and physical asynchronous client attempts in the existing PostgreSQL database before calling a provider. Completion and outbox insertion share a transaction. The HTTP worker acknowledges exact event ID, version and digest triples after the receiver commits its inbox. Network failures, missing acknowledgements, 429 and 5xx retain pending events. Rejections and content conflicts retain their evidence and emit an error log

The canonical protocol belongs to the GuangRouter repository under `doc/contracts/litellm-usage-v1`. `protocol.py`, `contracts/batch.schema.json` and `contracts/examples.json` must produce the same schema and digests in both repositories

## Migration and activation

Build from `stable/gr-v1.96.2` at `d9047c1d`. Prepare a traceable fork image using the dependency build workflow before ci-deploy consumes it. Do not build or patch LiteLLM on production hosts

First land the compatible GuangRouter receiver and drain every older accounting writer. Run this command with the existing LiteLLM database URL supplied through the deployment secret environment. The required database name is a target check. Omitting `--apply` is a preview

```sh
python -m litellm.proxy.usage_journal.migrate --database litellm
python -m litellm.proxy.usage_journal.migrate --database litellm --apply
python -m litellm.proxy.usage_journal.migrate --database litellm
```

The migration takes a transaction advisory lock and records its operation ID and timestamp. It creates only the independent `litellm_usage` schema. Existing Prisma models and spend logs are unchanged. Any unexpected pre-existing schema or revision stops the operation. A failed transaction leaves the database unchanged; repeat the preview and apply after resolving its cause. Retain this schema on image rollback; never reset usage journals or financial records

Set `LITELLM_USAGE_REPORT_URL` to the internal GuangRouter `/api/internal/litellm/usage-reports` entrypoint and `LITELLM_USAGE_REPORT_TOKEN` to the existing internal callback bearer credential. Startup checks migration version 1 and starts one lease-based dispatcher per process. It never performs DDL. The relay authenticates its protocol headers with the same credential in `X-Guangrouter-Usage-Token`. The capability endpoint is `GET /health/usage-protocol`, authenticated with Bearer; absent support must refuse admission instead of silently using legacy accounting

Enable relay shadow reports first. Shadow events remain independent of customer accounting; compare their attempt/log identities, token counts and completion membership against spend logs. Only after A/B drain and comparison may the relay mark new requests `active`. Disabling the relay flag stops new protocol admissions while existing journals, callbacks and settlements continue draining

## Failure and resource boundaries

The default dispatch tick is 50 ms and each claim contains at most 100 events. HTTP runs outside database transactions. Claims expire after 30 seconds, permitting process or A/B takeover. Request identity uniqueness refuses provider replay, including after a crash. Lost processes produce unknown attempt evidence and a final membership declaration; they never recall a provider

Persistence or capacity failure before a provider attempt refuses the call. Already active streams retain their iterator and complete independently of delivery availability. Completion persistence retries retain their evidence in process; after a crash, the already committed journal remains unknown until recovered with actual evidence. Unknown usage never becomes zero

`LITELLM_USAGE_MAX_PENDING` defaults to 100000 and bounds pending events, admitted open requests and unresolved attempts. `LITELLM_USAGE_MAX_BYTES` defaults to 10 GiB across journal/outbox relations. Size it for the intended 24-hour acknowledged-body window and retained unresolved evidence. Hitting either limit stops new attempts. Recovery is to restore delivery/database service or explicitly increase provisioned capacity after inspecting the backlog; it never deletes unresolved or conflicting evidence

Acknowledged bodies are cleared in batches of 1000 only after 24 hours and only for closed requests with all usage known and no conflict. Event identity and digest audit records remain. Unknown, unacknowledged and conflicted request evidence is retained

## Verification

Run `make bootstrap`, then migrate an isolated test PostgreSQL database. Run the journal tests with `LITELLM_USAGE_TEST_DATABASE_URL` pointing at that database; do not share it with a running dispatcher. Existing common request processing tests also exercise the unmarked compatibility path

For the fork target, run `make pre-commit LINT_BASE_REMOTE=hex LINT_BASE_BRANCH=stable/gr-v1.96.2`. The default lint base remains the upstream staging branch

Release acceptance additionally requires 100 completed requests per second for ten minutes in both a single pool and multiple pools, ten browser pages polling list and detail every two seconds, P95 completion-to-accounting-commit at most two seconds, exact per-request accounting, no growing backlog, and real-provider checks across all five supported entrypoints. Record unavailable provider capabilities and intermediate migration, LAN/public cutover, drain and rollback evidence separately; local synthetic success cannot claim those production states were exercised
