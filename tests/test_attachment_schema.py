from __future__ import annotations

import json
from pathlib import Path

import jsonschema  # type: ignore[import-untyped, unused-ignore]


def _schema(name: str) -> dict[str, object]:
    schema_path = Path(__file__).resolve().parents[1] / "docs" / "schemas" / name
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_run_envelope_schema_accepts_sessions_and_attachments() -> None:
    schema = _schema("stormlog_run_envelope_v1.schema.json")
    payload = {
        "schema_version": 1,
        "format": "stormlog.run_envelope",
        "run_id": "run-train-42",
        "title": "Training run 42",
        "description": "CUDA training investigation",
        "job_id": "job-42",
        "started_at_ns": 1_800_000_000_000_000_000,
        "ended_at_ns": 1_800_000_060_000_000_000,
        "created_at_utc": "2026-06-14T21:00:00Z",
        "updated_at_utc": "2026-06-14T21:05:00Z",
        "source_namespace": "wandb",
        "source_ref": "entity/project/run-42",
        "tags": ["training", "cuda"],
        "sessions": [
            {
                "session_id": "session-rank-0",
                "job_id": "job-42",
                "rank": 0,
                "local_rank": 0,
                "world_size": 2,
                "role": "rank",
                "source_namespace": "stormlog",
                "source_ref": "rank-0",
                "metadata": {"host": "host-a"},
            }
        ],
        "attachments": [
            {
                "attachment_id": "profiler-trace",
                "title": "Profiler trace",
                "kind": "profiler_trace",
                "storage": "reference",
                "path": "traces/rank0.trace",
                "run_id": "run-train-42",
                "session_id": "session-rank-0",
                "job_id": "job-42",
                "rank": 0,
                "local_rank": 0,
                "world_size": 2,
                "start_ns": 1_800_000_000_000_000_000,
                "end_ns": 1_800_000_060_000_000_000,
                "created_at_utc": "2026-06-14T21:00:00Z",
                "updated_at_utc": "2026-06-14T21:05:00Z",
                "source_namespace": "nsys",
                "source_ref": "rank0",
                "metadata": {"tool": "nsys"},
            }
        ],
        "metadata": {"owner": "training"},
    }

    jsonschema.Draft202012Validator(schema).validate(payload)


def test_run_envelope_schema_rejects_missing_metadata() -> None:
    schema = _schema("stormlog_run_envelope_v1.schema.json")
    payload = {
        "schema_version": 1,
        "format": "stormlog.run_envelope",
        "run_id": "run-missing-metadata",
    }

    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(payload))

    assert errors


def test_attachment_schema_rejects_null_targets() -> None:
    schema = _schema("stormlog_attachments_v1.schema.json")
    payload = {
        "schema_version": 1,
        "format": "stormlog.attachments",
        "attachments": [
            {
                "title": "Profiler trace",
                "kind": "profiler",
                "url": None,
                "metadata": {},
            }
        ],
    }

    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(payload))

    assert errors


def test_attachment_schema_accepts_run_metadata_compatibly() -> None:
    schema = _schema("stormlog_attachments_v1.schema.json")
    payload = {
        "schema_version": 1,
        "format": "stormlog.attachments",
        "attachments": [
            {
                "title": "W&B run",
                "kind": "experiment",
                "url": "https://wandb.ai/example/project/runs/run-123",
                "run_id": "run-123",
                "session_id": "session-123",
                "job_id": "job-42",
                "rank": 0,
                "storage": "reference",
                "source_namespace": "wandb",
                "source_ref": "example/project/run-123",
                "metadata": {},
            }
        ],
    }

    jsonschema.Draft202012Validator(schema).validate(payload)
