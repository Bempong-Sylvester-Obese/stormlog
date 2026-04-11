"""Attribution-specific W&B export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..cuda_native_debug import (
    ALLOCATION_ATTRIBUTION_FILENAME,
    DEBUG_METADATA_FILENAME,
    TENSOR_ATTRIBUTION_FILENAME,
    TRACE_HTML_ANNOTATED_FILENAME,
    TRACE_HTML_FILENAME,
)
from .core import read_json_if_exists


def log_attribution_outputs(
    wandb: Any,
    run: Any,
    *,
    root: Path,
    session_slug: str,
) -> dict[str, Any]:
    summary_fields: dict[str, Any] = {}
    files_to_attach = [
        root / TRACE_HTML_ANNOTATED_FILENAME,
        root / TRACE_HTML_FILENAME,
        root / TENSOR_ATTRIBUTION_FILENAME,
        root / ALLOCATION_ATTRIBUTION_FILENAME,
        root / DEBUG_METADATA_FILENAME,
    ]
    existing_files = [
        path for path in files_to_attach if path.exists() and path.is_file()
    ]

    if existing_files:
        artifact = wandb.Artifact(
            name=f"stormlog-attribution-{session_slug}",
            type="stormlog-attribution",
        )
        for path in existing_files:
            artifact.add_file(local_path=str(path), name=path.name)
        run.log_artifact(artifact)

    html_path = root / TRACE_HTML_ANNOTATED_FILENAME
    if html_path.exists():
        run.log(
            {
                "stormlog_attribution_html": wandb.Html(
                    html_path.read_text(encoding="utf-8")
                )
            }
        )
        summary_fields["stormlog_attribution_html_file"] = html_path.name

    tensor_rows = tensor_attribution_rows(root / TENSOR_ATTRIBUTION_FILENAME)
    if tensor_rows:
        run.log(
            {
                "stormlog_tensor_attribution": wandb.Table(
                    columns=[
                        "name",
                        "storage_ptr",
                        "tensor_count",
                        "total_size_bytes",
                        "shape",
                        "dtype",
                    ],
                    data=tensor_rows[:200],
                )
            }
        )
        summary_fields["stormlog_tensor_attribution_rows"] = len(tensor_rows)

    metadata = read_json_if_exists(root / DEBUG_METADATA_FILENAME)
    if isinstance(metadata, Mapping):
        history_recorded = metadata.get("history_recorded")
        if isinstance(history_recorded, bool):
            summary_fields["stormlog_attribution_history_recorded"] = history_recorded

    return summary_fields


def tensor_attribution_rows(path: Path) -> list[list[Any]]:
    payload = read_json_if_exists(path)
    if not isinstance(payload, Mapping):
        return []
    entries = payload.get("attributed_storage_pointers")
    if not isinstance(entries, list):
        return []

    rows: list[list[Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        tensors = entry.get("tensors")
        if not isinstance(tensors, list):
            tensors = []
        total_size = 0
        shape = ""
        dtype = ""
        if tensors:
            first_tensor = tensors[0] if isinstance(tensors[0], Mapping) else {}
            shape = str(first_tensor.get("shape", ""))
            dtype = str(first_tensor.get("dtype", ""))
            for tensor in tensors:
                if isinstance(tensor, Mapping):
                    size_bytes = tensor.get("size_bytes", 0)
                    if isinstance(size_bytes, int):
                        total_size += size_bytes

        names = entry.get("names")
        name = "<unnamed>"
        if isinstance(names, list) and names:
            name = ", ".join(str(value) for value in names[:3])

        rows.append(
            [
                name,
                str(entry.get("storage_ptr", "")),
                int(entry.get("tensor_count", len(tensors))),
                total_size,
                shape,
                dtype,
            ]
        )
    rows.sort(key=lambda row: int(row[3]), reverse=True)
    return rows
