[← Back to docs](index.md)

# Run Envelopes

Stormlog uses sessions and run envelopes at different investigation levels:

- A **session** is one capture lifecycle from one tracker, TUI live capture,
  sink session, or diagnose bundle. It owns telemetry identity and lifecycle
  state such as `running`, `completed`, `interrupted`, or `incomplete`.
- A **run envelope** is the top-level investigation object. It can group one
  session, distributed rank-local sessions, local artifact bundles, copied
  files, and external references such as profiler traces or experiment links.

Run envelopes are local-first. Stormlog discovers `stormlog_run.json` files
from artifact directories, but it can still synthesize run rows from existing
session metadata when no envelope exists.

## `stormlog_run.json`

The v1 schema is `docs/schemas/stormlog_run_envelope_v1.schema.json`.

Required fields:

- `schema_version`: `1`
- `format`: `stormlog.run_envelope`
- `run_id`: stable run identifier
- `metadata`: integration-specific object, empty when no extra metadata exists

Optional top-level fields include `title`, `description`, `job_id`,
`started_at_ns`, `ended_at_ns`, `created_at_utc`, `updated_at_utc`,
`source_namespace`, `source_ref`, `tags`, `sessions`, and `attachments`.

Minimal example:

```json
{
  "schema_version": 1,
  "format": "stormlog.run_envelope",
  "run_id": "run-train-42",
  "job_id": "train-42",
  "sessions": [
    {
      "session_id": "session-rank-0",
      "job_id": "train-42",
      "rank": 0,
      "local_rank": 0,
      "world_size": 8,
      "role": "rank",
      "metadata": {}
    }
  ],
  "attachments": [
    {
      "attachment_id": "rank0-profiler",
      "title": "Rank 0 profiler trace",
      "kind": "profiler_trace",
      "storage": "reference",
      "path": "traces/rank0.trace",
      "session_id": "session-rank-0",
      "job_id": "train-42",
      "rank": 0,
      "metadata": {"tool": "nsys"}
    }
  ],
  "metadata": {}
}
```

Relative attachment paths resolve against the directory containing
`stormlog_run.json`. Stormlog records remote URLs during discovery but does not
fetch them.

## Discovery Rules

Explicit envelopes win when present. If a directory contains one or more
`stormlog_run.json` files, `stormlog.query` lists those explicit runs and uses
their session membership for run attachment indexing.

When no explicit envelope exists, Stormlog synthesizes run rows from sessions:

- rank-local sessions with the same non-null `job_id` become one distributed run
- sessions without `job_id` become one run each
- reusing a sink directory across chronological sessions does not merge them

This keeps existing artifact layouts compatible while giving future integrations
a stable run-level target.

## Attachments

Run attachments can represent:

- local telemetry files or append-only sink directories
- sink segments and rollup sidecars
- diagnose bundles and OOM bundles
- `stormlog_attachments.json` sidecar entries
- external references such as profiler URLs or experiment-tracking runs

`storage` is either `reference` or `copy`. References point to external systems
or local paths without copying bytes during discovery. Copies identify files
that are already part of the local artifact set.

`source_namespace` and `source_ref` are intentionally shallow. Use values such
as `wandb` / `entity/project/run-id` or `nsys` / `rank0` for navigation, and put
integration-specific details in `metadata`.

## Query API

Use the CLI:

```bash
stormlog query runs ./artifacts --json
stormlog query attachments ./artifacts --run-id run-train-42 --table
```

Or the Python API:

```python
import stormlog.query

store = stormlog.query.open(["./artifacts"])
runs = store.list_runs()
attachments = store.list_run_attachments()
```
