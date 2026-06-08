## Summary

- Added `rollups.json` v1 sidecars for append-only telemetry sinks so long-running monitor sessions can be browsed through compact derived summaries while raw JSONL remains authoritative.
- Implemented typed rollup dataclasses, deterministic JSON serialization, schema validation, atomic writes, stale/malformed read tolerance, and sink close/recovery generation.
- Added query-layer fast paths that use fresh sink rollups for exact built-in summaries and fall back to raw events otherwise.
- Documented the sidecar schema, compute timing, retention semantics, TUI consumption decision, write/storage overhead estimates, and follow-on tasks.

## Design Decisions

- Rollups are computed on sink close and recovery, not append or periodic flush, to keep the capture hot path unchanged.
- `rollups.json` is a derived sidecar and can be deleted or rebuilt from retained JSONL segments.
- Sidecar freshness is tied to retained segment filenames, event counts, bytes, and manifest schema version.
- Query consumption is conservative: only summaries that can be answered exactly from a fresh sidecar use it.

## Validation

- `.venv/bin/python -m pytest tests/test_query.py tests/test_telemetry_sink.py tests/test_telemetry_rollups.py`
- Commit hooks run per checkpoint: `isort`, `black`, `flake8`, trailing whitespace, end-of-file, large-file check, and `mypy`.

## Follow-On Tasks

- Add a post-processing CLI to rebuild `rollups.json` for existing sinks.
- Teach TUI overview/session comparison panels to prefer fresh rollups before raw replay.
- Benchmark rebuild time and storage overhead on large retained sink directories.
- Keep persisted issue state in a separate future sidecar rather than mixing issue workflow state into rollups.
