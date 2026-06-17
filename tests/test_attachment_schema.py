from __future__ import annotations

import json
from pathlib import Path

import jsonschema  # type: ignore[import-untyped, unused-ignore]


def test_attachment_schema_rejects_null_targets() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "schemas"
        / "stormlog_attachments_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
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
