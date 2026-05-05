[← Back to main docs](index.md)

# Stormlog Telemetry-First Internal Model

Stormlog uses a telemetry-first internal projection to give runtime capture,
append-only sink persistence, artifact loading, live display, and offline
analysis one shared event model.

The existing `TelemetryEvent v3` format remains the stable artifact and sink
format. The backend-neutral model is an internal projection over that format,
implemented in `stormlog.telemetry_model` and exposed through
`stormlog.telemetry`.

## Implemented Model

The canonical projection is `CanonicalTelemetryRecord`. It is a small immutable
event envelope with these fields:

- `schema_version`: internal canonical projection version.
- `record_id`: deterministic identifier derived from the source telemetry
  record.
- `timestamp_ns`: event time from the source.
- `observed_timestamp_ns`: time Stormlog observed or normalized the event.
- `session_id`: capture/session identity.
- `source_kind`: backend family such as `cpu`, `cuda`, `rocm`, `mps`,
  `tensorflow`, `tpu`, or `other`.
- `event_type`: generic classification such as `sample`, `start`, `stop`,
  `phase_enter`, `phase_exit`, `warning`, `critical`, or `error`.
- `stage`: optional lifecycle or workload stage.
- `severity` and `severity_text`: normalized severity when meaningful.
- `body`: primary message or payload.
- `resource`: runtime identity such as host, process, backend, device, job, and
  rank.
- `attributes`: extensible metadata and backend-specific measurements.
- `correlation`: session, job, rank, phase, and future trace/span alignment
  fields.

The projection keeps backend-specific details out of top-level fields. Memory
counters from `TelemetryEvent v3`, collector health metadata, phase metadata,
and future backend details are represented as attributes, resources, or
correlation fields.

## Data Flow

The current flow is:

1. Trackers capture compact runtime events or tracker-local records.
2. Existing normalizers produce `TelemetryEvent v3` records for exports, sink
   writes, loaders, and TUI adapters.
3. `canonical_record_from_telemetry_event(...)` projects V3 records into
   `CanonicalTelemetryRecord`.
4. `LoadedTelemetrySession.telemetry_records()` exposes canonical records for
   loaded artifacts.
5. `TrackerSession.telemetry_records()` exposes canonical records for live TUI
   sessions.

This keeps the capture path cheap while giving downstream code one stable
backend-neutral view.

## Compatibility

Stormlog preserves existing compatibility boundaries:

- Existing V2, V3, and legacy artifacts remain loadable.
- `TelemetryEvent v3` remains the persisted artifact and append-only sink
  record format.
- Legacy artifact upcasting stays in loader and normalizer code.
- Canonical projection is additive and does not change CLI, TUI, or Python API
  behavior.
- Legacy export shapes remain explicit compatibility paths.

This lets analysis and UI code adopt the canonical projection without breaking
older artifacts or changing the on-disk schema.

## Live and Loaded Sessions

Live and loaded data now share the same canonical record projection:

- Loaded artifacts use `LoadedTelemetrySession.telemetry_records()`.
- Loaded artifacts use `LoadedTelemetrySession.resources()` for unique observed
  resources.
- Loaded artifacts use `LoadedTelemetrySession.correlations()` for unique
  correlation contexts.
- Live TUI sessions use `TrackerSession.telemetry_records()`.

The TUI can keep rendering lightweight view models, while analysis and future
query code can use the canonical records regardless of whether the source is a
live tracker or an artifact.

## Performance Policy

Always-on capture protects application liveness first. The hot path should do
only the work required to capture local fields, timestamp events, look up
session/correlation identity, and append to bounded in-memory history or a
queue.

Capture code should avoid:

- blocking on sink persistence,
- heap-heavy serialization,
- reflection-heavy shaping,
- formatting strings before normalization when raw fields are enough.

Sink persistence should remain batched and append-only. Under pressure,
Stormlog should prefer explicit sampling or low-priority event shedding over
blocking the application being measured.

## Benchmark Plan

Benchmark validation should compare the existing V3 flow against the canonical
projection path before any persisted-format migration.

Measure:

- events per second sustained,
- average and p95 event capture latency,
- allocations per event,
- bytes allocated per second,
- queue or bounded-history depth under burst load,
- sink flush throughput,
- TUI update latency in live mode,
- artifact load time for large sessions,
- memory growth over long always-on runs.

Coverage should include:

- CPU-only workloads,
- GPU-heavy workloads,
- mixed backend workloads,
- quiet long-running always-on sessions,
- bursty error-heavy sessions.

The acceptance bar is that canonical projection must not materially harm the
hot path. Any extra projection cost must be offset by simpler shared sinks,
loaders, UI adapters, and analysis code.

## Migration Plan

### Phase 1: Canonical Projection

- Keep `TelemetryEvent v3` as the persisted artifact format.
- Project V3 records into `CanonicalTelemetryRecord`.
- Expose canonical records from loaded sessions and live TUI sessions.
- Cover projection behavior with deterministic tests.

### Phase 2: Tracker Adapters

- Keep tracker-local runtime events compact.
- Move PyTorch, CPU, and TensorFlow normalization into shared adapter helpers.
- Centralize timestamp rules, session identity, severity mapping, backend tags,
  collector health, and distributed correlation.
- Add capture and enqueue counters for benchmark visibility.

### Phase 3: Sink Migration

- Add a future sink schema version for persisted canonical records.
- Keep append-only JSONL semantics and deterministic serialization.
- Preserve rollover, pruning, recovery, and manifest behavior.
- Keep old sink loading paths for existing artifacts.

### Phase 4: Loader Migration

- Dispatch by artifact and sink schema version.
- Parse V2, V3, legacy JSON, JSONL, and future persisted canonical records into
  the same internal stream.
- Keep compatibility transforms separate from primary parsing.
- Preserve old fixtures through loader adapters instead of rewriting user data.

### Phase 5: TUI and Session Unification

- Render live monitoring and diagnostics from canonical records or derived view
  models built from canonical records.
- Keep TUI-specific formatting outside the core telemetry model.
- Reuse the same query and marker logic for live and offline sessions.

### Phase 6: Benchmarks and Regression Gates

- Extend the benchmark harness with capture-latency, allocation, queue-depth,
  sink-throughput, load-time, and TUI-latency metrics.
- Compare current V3 normalization against canonical projection and future
  persisted-canonical writes.
- Require regression and budget gates before enabling new persisted formats.

## Follow-On Tasks

- Tracker layer: introduce shared runtime-event-to-canonical adapter helpers for
  PyTorch, CPU, and TensorFlow paths.
- Sink layer: add canonical-record sink versioning while preserving append-only
  recovery and retention behavior.
- Loader layer: add version dispatch that upcasts V2, V3, and future persisted
  canonical records into one internal stream.
- TUI/session layer: move more live and loaded views onto canonical records.
- Benchmark layer: extend `examples.cli.benchmark_harness` and docs benchmark
  assets with overhead, allocation, queue-depth, load-time, and TUI-latency
  checks.
- Compatibility layer: add explicit legacy import/export adapters and document
  retirement criteria.

## Non-Goals

- Rewriting every tracker to emit fully normalized records immediately.
- Changing the existing `TelemetryEvent v3` JSON schema.
- Designing a new external protocol.
- Changing user-facing CLI, TUI, or Python API behavior.
- Optimizing for every future backend-specific field up front.
