[← Back to main docs](index.md)

# Stormlog Telemetry-First Internal Model

## Status

This document records the target direction for consolidating Stormlog around one primary telemetry event model across
runtime capture, append-only sink persistence, artifact loading, live display,
and offline analysis. It does not change the current runtime API,
`TelemetryEvent v3` schema, sink format, loader behavior, or TUI behavior.

## Decision

Stormlog should move toward one telemetry-first internal model for runtime
capture, append-only sink persistence, artifact loading, live display, and
offline analysis.

The preferred architecture is:

1. Trackers capture compact runtime events with only cheap local fields.
2. A shared adapter normalizes those events into canonical telemetry records as
   early as possible outside the hottest capture path.
3. Sinks persist canonical records only.
4. Loaders convert every supported artifact version into canonical records.
5. The TUI and analysis layers consume one session abstraction for live and
   loaded data.

This keeps tracker capture inexpensive while giving downstream code a stable
record shape for queryability, aggregation, integrations, and schema evolution.

## Current State

Stormlog already has several pieces of this architecture:

- `stormlog.telemetry.TelemetryEventV3` is the current canonical event used by
  tracker exports, append-only sink segments, synthesized diagnose timelines,
  and loader output.
- `stormlog.telemetry.TelemetryEventV2` and permissive legacy loading preserve
  older artifact compatibility.
- PyTorch and CPU trackers keep lightweight `TrackingEvent` instances, then
  convert them through `_telemetry_record_from_event(...)`.
- The TensorFlow tracker builds normalized records in
  `_build_telemetry_event_record(...)`.
- `AppendOnlyTelemetrySink` writes newline-delimited JSON records and tracks
  session summaries through its manifest.
- `load_telemetry_sessions(...)` loads JSON, JSONL, and sink directories into
  `LoadedTelemetrySession` objects grouped by `session_id`.
- `stormlog.tui.monitor.TrackerSession` adapts live tracker output into TUI
  view models and can expose normalized telemetry events.

The gap is that the current V3 shape is still memory-metric oriented. It works
well for today's CPU, CUDA, ROCm, MPS, and TensorFlow memory capture paths, but
future queryability and backend-neutral integrations need a smaller canonical
envelope with extensible attributes.

## Target Record Contract

The next canonical record should be small, immutable, versioned, and
backend-neutral. Required fields should cover identity, time, classification,
source context, and extensibility without hard-coding one runtime's memory
model.

Recommended fields:

- `schema_version`: canonical envelope version.
- `record_id`: stable unique identifier for the record.
- `timestamp_ns`: event time from the source, in nanoseconds.
- `observed_timestamp_ns`: time Stormlog observed or normalized the event.
- `session_id`: capture/session identity.
- `source_kind`: backend family such as `cpu`, `cuda`, `rocm`, `mps`,
  `tensorflow`, `tpu`, or `other`.
- `event_type`: generic event classification such as `sample`, `start`,
  `stop`, `phase_enter`, `phase_exit`, `warning`, `critical`, or `error`.
- `stage`: optional lifecycle or workload stage.
- `severity`: optional normalized severity value.
- `body`: primary message or payload for the event.
- `resource`: runtime identity such as host, process, job, worker, backend,
  device, framework, and version.
- `attributes`: extensible metadata for backend-specific values such as memory
  counters, stream identifiers, tensor shapes, batch sizes, driver versions, or
  collector health.
- `correlation`: run, trace, span, phase, rank, or parent identifiers used to
  connect records.

Memory-specific fields from `TelemetryEvent v3`, such as allocator counters,
device counters, and collector health, should become named attributes or
resource fields in the future envelope rather than top-level requirements.

## Compatibility Policy

Compatibility must stay explicit and boundary-local:

- Existing V2, V3, and legacy artifacts remain loadable.
- The V3 schema remains the current write format until a separate migration
  changes runtime behavior.
- Future new writes should use the new canonical envelope only after sink,
  loader, exporter, and TUI compatibility paths are ready.
- Legacy input adapters convert old tracker shapes or artifact versions into
  canonical records.
- Legacy output adapters project canonical records back into old export shapes
  only when callers explicitly request compatibility.
- Compatibility shims should live outside the core model and include removal
  criteria once supported artifact windows are old enough to retire.

Older artifacts should be handled with versioned loaders:

1. Detect artifact or sink schema version.
2. Parse records with the version-specific reader.
3. Upcast into canonical records.
4. Preserve source warnings when fields are missing or lossy.
5. Build indexes after loading unless streaming UX requires incremental
   indexing.

## Performance Policy

Always-on capture must protect application liveness first. The capture path
should avoid blocking I/O, reflection-heavy shaping, and heap-heavy
serialization.

The hot path should do only:

- field capture,
- timestamping,
- correlation/session lookup,
- bounded queue or history append,
- optional fixed-cost normalization when it is proven cheap.

Sink persistence should remain batched and asynchronous. Under pressure,
Stormlog should prefer explicit sampling or low-priority event shedding over
blocking the application being measured.

Benchmark acceptance should be based on:

- per-event capture latency,
- allocations per event,
- bytes allocated per second,
- queue or bounded-history depth under bursts,
- sink flush throughput,
- p95 and p99 capture impact,
- TUI update latency in live mode,
- artifact load time for large sessions.

The benchmark suite should compare the current split model against the future
canonical-envelope implementation before any runtime migration is accepted.

## Shared Session Interface

Live sessions and loaded artifacts should expose the same model to TUI and
analysis code. The backing source may differ, but consumers should see the same
query surface.

Target interface capabilities:

- `events()` returns canonical records.
- `resources()` returns observed resources and backend identities.
- `summary()` returns session lifecycle and high-level capture metadata.
- `filter(query)` returns records matching event, resource, attribute, or time
  predicates.
- `timeline(range)` returns ordered records and derived markers.
- `correlate(id)` returns related records for a run, phase, trace, rank, or
  span-like identifier.

Live implementations can be backed by a bounded async stream. Loaded
implementations can be backed by indexed artifact readers. Both should feed the
same TUI tables, timelines, summaries, and analysis code.

## Migration Plan

### Phase 1: Canonical Envelope

- Define the next canonical record type and schema version.
- Keep V3 as the current public artifact format until migration is ready.
- Add conversion helpers from V2, V3, and legacy records into the new envelope.
- Add deterministic serialization tests for the new envelope.

### Phase 2: Tracker Adapters

- Keep tracker-local runtime events compact.
- Move PyTorch, CPU, and TensorFlow normalization into shared adapter helpers.
- Centralize timestamp rules, session identity, severity mapping, backend tags,
  collector health, and distributed correlation.
- Add capture and enqueue counters for benchmark visibility.

### Phase 3: Sink Migration

- Add a sink schema version that persists canonical envelope records.
- Keep append-only JSONL semantics and deterministic serialization.
- Batch writes and preserve existing rollover, pruning, recovery, and manifest
  behavior.
- Keep old sink loading paths for existing artifacts.

### Phase 4: Loader Migration

- Dispatch by artifact and sink schema version.
- Parse V2, V3, legacy JSON, JSONL, and canonical envelope records into the same
  internal stream.
- Keep compatibility transforms separate from primary parsing.
- Preserve old fixtures through loader adapters instead of rewriting user data.

### Phase 5: TUI and Session Unification

- Introduce the shared session abstraction for live and loaded data.
- Render TUI monitoring and diagnostics from canonical records or derived view
  models built from canonical records.
- Keep TUI-specific formatting outside the core telemetry model.
- Reuse the same query and marker logic for live and offline sessions.

### Phase 6: Benchmarks and Regression Gates

- Extend the existing benchmark harness with capture-latency, allocation,
  queue-depth, sink-throughput, load-time, and TUI-latency metrics.
- Cover CPU-only, GPU-heavy, TensorFlow, mixed backend, quiet always-on, and
  bursty error-heavy sessions.
- Compare current V3 normalization against canonical-envelope normalization.
- Require regression and budget gates before enabling new runtime writes.

### Phase 7: Compatibility Sunset

- Keep V2/V3 loaders and explicit legacy exports until the supported artifact
  window expires.
- Document removal criteria for each shim.
- Remove old compatibility code only after fixtures, docs, and release notes
  prove that user artifacts remain protected.

## Follow-On Tasks

- Tracker layer: introduce shared runtime-event-to-canonical adapter helpers for
  PyTorch, CPU, and TensorFlow paths.
- Sink layer: add canonical-envelope sink versioning while preserving append-only
  recovery and retention behavior.
- Loader layer: add version dispatch that upcasts V2, V3, and future envelope
  records into one internal stream.
- TUI/session layer: define and adopt the shared live/loaded session query
  interface.
- Benchmark layer: extend `examples.cli.benchmark_harness` and docs benchmark
  assets with overhead, allocation, queue-depth, load-time, and TUI-latency
  checks.
- Compatibility layer: add explicit legacy import/export adapters and document
  retirement criteria.

## Non-Goals

- Rewriting every tracker to emit fully normalized records immediately.
- Changing the existing `TelemetryEvent v3` JSON schema in this issue.
- Designing a new external protocol.
- Changing user-facing CLI, TUI, or Python API behavior.
- Optimizing for every future backend-specific field up front.
