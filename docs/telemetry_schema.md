[← Back to main docs](index.md)

# TelemetryEvent v3 Schema

`TelemetryEvent v3` is the canonical event format for tracker exports,
append-only sink segments, synthesized diagnose timelines, and loader output.

Schema files:

- `docs/schemas/telemetry_event_v3.schema.json`
- legacy compatibility: `docs/schemas/telemetry_event_v2.schema.json`

## Required fields

- `schema_version` (`3`)
- `session_id`
- `timestamp_ns`
- `event_type`
- `collector`
- `sampling_interval_ms`
- `pid`
- `host`
- `device_id`
- `allocator_allocated_bytes`
- `allocator_reserved_bytes`
- `allocator_active_bytes`
- `allocator_inactive_bytes`
- `allocator_change_bytes`
- `device_used_bytes`
- `device_free_bytes`
- `device_total_bytes`
- `context`
- `metadata`

## Session identity and lifecycle

Every long-running capture now belongs to exactly one session.

- `session_id` is an opaque UUID string generated once per `track` run, TUI live session, or standalone `diagnose` bundle
- session identity is the authoritative grouping key for telemetry and attached artifacts
- host, job, and rank remain descriptive metadata, but they do not define capture ownership

Session lifecycle is recorded separately from the event stream:

- `running`: session started but no terminal state has been persisted yet
- `completed`: clean shutdown finished and terminal session state was written
- `interrupted`: a previously running append-only sink session was recovered on the next startup
- `incomplete`: loaders found partial or orphaned artifacts that cannot prove a clean or recovered stop

Default session selection when multiple sessions are present:

1. newest `completed`
2. newest `interrupted`
3. newest `incomplete`

## Distributed identity fields

`TelemetryEvent v3` also recognizes these top-level distributed identity fields:

- `job_id` (`string | null`)
- `rank` (`integer`)
- `local_rank` (`integer`)
- `world_size` (`integer`)

New exports always emit these fields. For single-process runs, the defaults are:

- `job_id` -> `null`
- `rank` -> `0`
- `local_rank` -> `0`
- `world_size` -> `1`

`TelemetryEvent v3` validation is strict:

- unknown top-level fields are rejected
- `metadata` must be a JSON object (`dict` in Python terms)
- `rank` and `local_rank` must be >= `0`
- `world_size` must be >= `1`
- `rank` and `local_rank` must be < `world_size`

## Collector values

- `stormlog.cuda_tracker`
- `stormlog.rocm_tracker`
- `stormlog.mps_tracker`
- `stormlog.cpu_tracker`
- `stormlog.tensorflow.memory_tracker`

## Backend capability metadata

Tracker exports may include backend capability hints under `metadata`:

- `backend`
- `supports_device_total`
- `supports_device_free`
- `sampling_source`

## Collector health metadata

Always-on tracker exports annotate collector degradation in `metadata`:

- `collector_health_status` (`healthy`, `degraded`, `unhealthy`)
- `telemetry_partial` (`bool`)
- `collector_partial_fields` (`list[str]`)
- `collector_last_error` (`string | null`)
- `collector_consecutive_failures` (`integer`)
- `collector_next_retry_epoch_s` (`float | null`)

Tracker exports may also emit these lifecycle events:

- `collector_degraded`
- `collector_recovered`
- `start`
- `stop`
- `phase_enter`
- `phase_exit`

When the collector cannot produce core metrics, Stormlog pauses sample emission
until the next retry window and records the degraded state instead of exporting
synthetic zero-valued samples.

## Structured phase metadata

Phase-aware trackers store workload boundaries under `metadata["phase_scope"]`.
This payload is attached to `phase_enter` and `phase_exit` events and is replayed
later by the analyzer and TUI.

The current shape is:

- `action` (`enter` or `exit`)
- `name` (leaf phase label)
- `path` (`list[str]`)
- `depth` (`int`)
- `scope_id` (`string`)
- `parent_scope_id` (`string | null`)
- `thread_id` (`int`)
- `thread_name` (`string`)
- `sequence` (`int`)
- `attributes` (`object`, optional)

Example:

```json
{
  "event_type": "phase_enter",
  "metadata": {
    "phase_scope": {
      "action": "enter",
      "name": "forward",
      "path": ["train", "forward"],
      "depth": 2,
      "scope_id": "session-1:2",
      "parent_scope_id": "session-1:1",
      "thread_id": 88,
      "thread_name": "MainThread",
      "sequence": 2,
      "attributes": {"epoch": 3}
    }
  }
}
```

## Analyzer phase attribution payloads

Analyze and Diagnostics outputs may also include a derived
`phase_attribution` object. This is not part of the raw event schema above; it
is report-layer data produced after replaying the `phase_scope` boundaries.

Canonical fields:

- `phase_resolution` (`unique`, `ambiguous`, or omitted when no attribution exists)
- `phase_source` (`exact`, `thread_local`, `heuristic`, or omitted)
- `phase_path` (`string`, only for unique attributions)
- `phase_paths` (`list[str]`, one or more candidate labels)
- `scope_id`, `thread_id`, `thread_name` (present only when uniquely tied to one scope)

Optional presentation field:

- `phase_summary`
  - `phase_path`
  - `source`

`phase_summary` is emitted only when the product wants one useful display label
even though the canonical attribution remains ambiguous. For example, the CLI
may show `(likely) train / communication` while the underlying
`phase_attribution.phase_resolution` still stays `ambiguous`.

If ambiguity collapses to only one distinct label, Stormlog keeps the canonical
ambiguity and omits `phase_summary` because there is no materially different
winner to show.

## Legacy compatibility

Legacy conversion is permissive by default in
`stormlog.telemetry.telemetry_event_from_record`. Legacy conversion is attempted
only when `schema_version` is absent.

If `schema_version` is present:

- it must be an integer
- it must be a supported schema version
- any other value is rejected without legacy fallback

Legacy defaults:

- missing `pid` -> `-1`
- missing `host` -> `"unknown"`
- missing `device_id` -> inferred from `device` if possible, otherwise `-1`
- missing `allocator_reserved_bytes` -> `allocator_allocated_bytes`
- missing `allocator_change_bytes` -> `0`
- missing `device_used_bytes` -> `allocator_allocated_bytes`
- missing `device_total_bytes` and `device_free_bytes` -> `null`
- missing `event_type` -> `type` field if present, else `"sample"`
- missing distributed identity -> `job_id: null`, `rank: 0`, `local_rank: 0`, `world_size: 1`
- legacy `metadata_*` fields are folded into the canonical `metadata` object
- legacy artifacts without `session_id` receive a deterministic synthetic session id during load

If a legacy record is missing a valid timestamp, conversion fails.

## Distributed env inference

Tracker constructors can infer distributed identity from common launcher env vars:

- PyTorch / `torchrun`: `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `TORCHELASTIC_RUN_ID`
- Open MPI: `OMPI_COMM_WORLD_RANK`, `OMPI_COMM_WORLD_LOCAL_RANK`, `OMPI_COMM_WORLD_SIZE`
- Slurm: `SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NTASKS`, `SLURM_JOB_ID`

CLI and Python API callers can override these values explicitly.

## Python API

Use the public conversion, validation, and loading helpers in `stormlog.telemetry`:

```python
from stormlog.telemetry import (
    load_telemetry_events,
    load_telemetry_sessions,
    telemetry_event_from_record,
    telemetry_event_to_dict,
    validate_telemetry_record,
)
```

- `load_telemetry_sessions(path, permissive_legacy=True, events_key=None)`
- `load_telemetry_events(path, permissive_legacy=True, events_key=None, session_id=None)`
- `telemetry_event_from_record(record, permissive_legacy=True, ...)`
- `validate_telemetry_record(record)`

These APIs normalize legacy records to canonical `schema_version: 3` events and
enforce required fields.

## Append-only sink layout

Always-on `track` sessions can also write append-only telemetry into a sink
directory during the run:

```text
telemetry_sink/
  manifest.json
  segment-000001.jsonl
  segment-000002.jsonl
```

- each JSONL line is one canonical telemetry record
- `manifest.json` is schema `v2` and tracks segment ordering, per-segment `session_id`, and a session ledger
- closed segments may be pruned when file-count or total-size retention limits are hit
- on recovery, previously running sessions are closed as `interrupted` and new writes start in a new session

`load_telemetry_events()` can read:

- a normal JSON export
- a sink directory
- `manifest.json`
- an individual JSONL segment

If the final JSONL line is truncated because the process was interrupted during a
write, the loader ignores that incomplete tail and still returns the fully
written records ahead of it. If the artifacts cannot prove recovered ownership,
the loader classifies the older capture as `incomplete`.

## Diagnose and OOM manifests

Standalone `diagnose` bundles now write manifest schema `v2` and include:

- `session_id`
- `session_status`
- `session`

OOM flight-recorder bundles now write manifest schema `v2` and metadata that
reference the owning tracking `session_id` directly. That makes it possible to
tie a bundle back to the exact capture that emitted the OOM.

## Reconstructing a capture

```python
from stormlog.telemetry import load_telemetry_events, load_telemetry_sessions

sessions = load_telemetry_sessions("./live_sink")
selected = sessions[0]
events = load_telemetry_events(
    "./live_sink",
    session_id=selected.summary.session_id,
)

print(selected.summary.session_id)
print(selected.summary.status)
print(len(events))
```
