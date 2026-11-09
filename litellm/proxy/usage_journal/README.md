# Durable usage protocol v1

This fork journals admitted requests and physical asynchronous client attempts in the existing PostgreSQL database before calling a provider. Completion and outbox insertion share a transaction. The HTTP worker acknowledges exact event ID, version and digest triples after the receiver commits its inbox. Network failures, missing acknowledgements, 429 and 5xx retain pending events. Rejections and content conflicts retain their evidence and emit an error log

The canonical protocol belongs to the GuangRouter repository under `doc/contracts/litellm-usage-v1`. `protocol.py`, `contracts/batch.schema.json` and `contracts/examples.json` must produce the same schema and digests in both repositories

## Migration and activation

Build the frozen reviewed head of `stable/gr-v1.96.2`. Prepare a traceable fork image using the dependency build workflow before ci-deploy consumes it. Do not build or patch LiteLLM on production hosts

First land the compatible GuangRouter receiver and drain every older accounting writer. Run this command with the existing LiteLLM database URL supplied through the deployment secret environment. The required database name is a target check. Omitting `--apply` is a preview

```sh
python -m litellm.proxy.usage_journal.migrate --database litellm
python -m litellm.proxy.usage_journal.migrate --database litellm --apply
python -m litellm.proxy.usage_journal.migrate --database litellm
```

The migration takes a transaction advisory lock and records its operation ID and timestamp. It creates only the independent `litellm_usage` schema. Existing Prisma models and spend logs are unchanged. Any unexpected pre-existing schema or revision stops the operation. A failed transaction leaves the database unchanged; repeat the preview and apply after resolving its cause. Retain this schema on image rollback; never reset usage journals or financial records

Set `LITELLM_USAGE_REPORT_URL` to the internal GuangRouter `/api/internal/litellm/usage-reports` entrypoint and `LITELLM_USAGE_REPORT_TOKEN` to the existing internal callback bearer credential. Startup checks base migration 1 and additive feature migration 2 and starts one lease-based dispatcher per process. It never performs DDL. The relay authenticates its protocol headers with the same credential in `X-Guangrouter-Usage-Token`. The capability endpoint is `GET /health/usage-protocol`, authenticated with Bearer; absent support must refuse admission instead of silently using legacy accounting

Enable relay shadow reports first. Shadow events remain independent of customer accounting; compare their attempt/log identities, token counts and completion membership against spend logs. Only after A/B drain and comparison may the relay mark new requests `active`. Disabling the relay flag stops new protocol admissions while existing journals, callbacks and settlements continue draining

## Failure and resource boundaries

Journal usage comes from supplier HTTP response bytes before SDK conversion. The default shared HTTP and OpenAI SDK clients capture JSON counts or terminal SSE usage without changing customer responses. SDK zero defaults, estimated stream counts and outgoing proxy chunks cannot prove supplier usage. Missing evidence stays unknown, including after a client disconnect. Supplier-declared zero counts remain known

Streaming capture passes through the original bytes and recognizes identity, gzip and deflate encodings. Non-SSE streams remain unread by the capture hook. Unsupported stream formats, encodings and caller-supplied clients without the response hook leave usage unknown. Spend snapshots retain the captured usage and its cost provenance so recovery cannot promote SDK estimates or defaults. Older snapshots without supplier usage provenance remain unresolved rather than inventing a charge

The active dispatch tick is 50 ms; empty polling exponentially backs off to one second. Each claim is ordered by its due-time index, and contains at most 100 events. HTTP runs outside database transactions. Claims expire after 30 seconds, permitting process or A/B takeover. Request identity uniqueness refuses provider replay, including after a crash. Lost processes produce unknown attempt evidence and a final membership declaration; they never recall a provider

Each producer permits four concurrent journal transactions. Atomic SQL statements combine row changes, outbox insertion and capacity-counter changes while retaining transaction timeouts and request/attempt fencing. The customer-journal start/attempt/finish/close transactions have a regression budget of twenty Prisma HTTP round trips. Default Prisma HTTP clients retain at most twenty idle connections while keeping the existing active-connection limit and caller-supplied initial/reconnect settings

Startup verifies both journal migrations before publishing the producer instance. Authenticated capability probes recheck the live journal and report its instance and the cached contract digest. An unavailable database refuses capability before the relay reserves budget. Every inference admission additionally requires a committed journal transaction before provider I/O

Delivery and periodic maintenance run in separate supervised tasks. A blocked recovery pass cannot pause delivery; cancellation joins both tasks before closing the HTTP client. Completion schedules application of an already retained supplier snapshot without waiting for supplier observation work. A capture that began before completion refreshes the attempt after retaining its source, covering either commit order. Missing or failed supplier evidence remains eligible for the bounded periodic recovery pass

Persistence or capacity failure before a provider attempt refuses the call. Already active streams retain their iterator and complete independently of delivery availability. Completion persistence retries retain their evidence in process; after a crash, the already committed journal remains unknown until recovered with actual evidence. Unknown usage never becomes zero

`LITELLM_USAGE_MAX_PENDING` defaults to 100000 and bounds pending events, admitted open requests and unresolved attempts. `LITELLM_USAGE_MAX_BYTES` defaults to 10 GiB across journal/outbox relations. Size it for the intended 24-hour acknowledged-body window and retained unresolved evidence. Hitting either limit stops new attempts. Recovery is to restore delivery/database service or explicitly increase provisioned capacity after inspecting the backlog; it never deletes unresolved or conflicting evidence

Acknowledged bodies are cleared in batches of 1000 only after 24 hours and only for closed requests with all usage known and no conflict. Event identity and digest audit records remain. Unknown, unacknowledged and conflicted request evidence is retained

## Verification

Run `make bootstrap`, then migrate an isolated test PostgreSQL database. Run the journal tests with `LITELLM_USAGE_TEST_DATABASE_URL` pointing at that database; do not share it with a running dispatcher. Existing common request processing tests also exercise the unmarked compatibility path

For the fork target, run `make pre-commit LINT_BASE_REMOTE=hex LINT_BASE_BRANCH=stable/gr-v1.96.2`. The default lint base remains the upstream staging branch

Release acceptance additionally requires 100 completed requests per second for ten minutes in both a single pool and multiple pools, ten browser pages polling list and detail every two seconds, P95 completion-to-accounting-commit at most two seconds, exact per-request accounting, no growing backlog, and real-provider checks across all five supported entrypoints. Record unavailable provider capabilities and intermediate migration, LAN/public cutover, drain and rollback evidence separately; local synthetic success cannot claim those production states were exercised


## Readiness for retiring spend replay

The protocol remains `litellm-usage/1`; it had no production admissions before this contract was completed. The capability response includes `contractDigest`, the SHA-256 of canonical sorted JSON Schema. The relay sends the matching `X-GuangRouter-Usage-Contract` header on admission, and the dispatcher sends it on delivery. Mismatches refuse before provider I/O or inbox acknowledgement. Both admission flags remain explicitly disabled by default; unsupported gateways are not silently downgraded after activation

`attempt.observed` is independent of the immutable customer usage snapshot. It carries the native recorded USD cost (or an explicit unknown), deployment, timing and supplier-attempt outcome. Its event identity is `<requestId>/observation/<attemptId>`. Unknown cost may be enriched; contradictory known cost is retained as a conflict. Customer charges do not depend on an invented supplier price

The native spend writer retains a bounded metadata-only source snapshot before returning. Recovery claims include closed requests with incomplete attempts, so an in-memory completion retry is never the only recovery path. Every ten seconds a bounded leased pass looks up the retained snapshot or the exact `LiteLLM_SpendLogs.request_id`; it does not scan time windows or recall a provider. Missing evidence stays unknown with bounded exponential retry and an error/diagnostic trail. Source records with zero default counters do not establish known zero usage without an explicit usage object

Open requests are fenced to the recovering instance before their final attempt membership is sealed. A recovered unknown completion has no invented end timestamp or provider failure. Late actual evidence can enrich it, while known usage and customer accounting cannot be restated. Generic `workbench_model_control` callbacks are suppressed only inside authenticated active journal contexts; shadow and unmarked requests retain their original callbacks. Other callback integrations are unaffected

Feature migration 002 adds independent recovery state and leaves the original base revision readable by the previous producer. Preview and apply use the same explicit database target and advisory guard. A version-1 journal with more than 1000 existing attempts is refused before modification; inspect and prepare a separately bounded seed operation for that unexpected state. Known matching migrations repeat with `changed:false`; unexpected tables, feature columns or indexes stop the run. Rollback retains all journal tables and evidence

Shipping this PR does not disable the GuangRouter spend poller. The rollout remains receiver first, migration before candidate start, flags off during old/new coexistence, shadow comparison, then an explicit active decision after old incompatible accounting writers retire. After active admission, the minimum receiver rollback version must understand this exact contract digest. Restore routes/images without downgrading shared schemas or resetting accounting. Lost/unknown evidence and supplier discrepancies require targeted recovery and independent reconciliation even after periodic replay is eventually retired

Recorded zero supplier cost requires the journal logging provenance marker. A legacy spend-log default of zero without that evidence remains unknown; it does not complete supplier recovery or create a zero-cost quota event
