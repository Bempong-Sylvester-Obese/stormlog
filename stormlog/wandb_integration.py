"""Optional Weights & Biases export helpers for Stormlog outputs."""

from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cuda_native_debug import (
    TENSOR_ATTRIBUTION_FILENAME,
    TRACE_HTML_ANNOTATED_FILENAME,
)
from .session import SessionSummary, session_summary_from_dict

WANDB_INSTALL_GUIDANCE = (
    "Weights & Biases integration requires optional dependencies. "
    "Install with `pip install 'stormlog[wandb]'`."
)
_ALERT_EVENT_TYPES = frozenset({"warning", "critical", "error", "peak"})
_TIMELINE_MAX_POINTS = 250
_DASHBOARD_WIDTH = 720
_DASHBOARD_HEIGHT = 260
_DASHBOARD_PADDING = 18


@dataclass(frozen=True)
class WandbExportConfig:
    """Runtime configuration for optional W&B exports."""

    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    mode: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    group: str | None = None
    job_type: str | None = None
    log_tables: bool = True
    log_artifacts: bool = False
    log_attribution: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {None, "online", "offline"}:
            raise ValueError("wandb mode must be 'online', 'offline', or omitted")


def wandb_config_from_namespace(args: Any) -> WandbExportConfig:
    """Build a W&B export config from CLI args or a similar namespace."""
    return WandbExportConfig(
        enabled=bool(getattr(args, "wandb", False)),
        project=_normalized_optional_string(getattr(args, "wandb_project", None)),
        entity=_normalized_optional_string(getattr(args, "wandb_entity", None)),
        mode=_normalized_optional_string(getattr(args, "wandb_mode", None)),
        run_id=_normalized_optional_string(getattr(args, "wandb_run_id", None)),
        run_name=_normalized_optional_string(getattr(args, "wandb_name", None)),
        group=_normalized_optional_string(getattr(args, "wandb_group", None)),
        job_type=_normalized_optional_string(getattr(args, "wandb_job_type", None)),
        log_artifacts=bool(getattr(args, "wandb_log_artifacts", False)),
        log_attribution=bool(getattr(args, "wandb_log_attribution", False)),
    )


def add_wandb_arguments(parser: Any) -> None:
    """Attach shared optional W&B flags to a CLI parser."""
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log Stormlog summaries to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (default: stormlog)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity or team name",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline"],
        default=None,
        help="W&B logging mode (default: online)",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Existing W&B run id to resume or attach to",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Explicit W&B run name",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="W&B group override (default: Stormlog job id)",
    )
    parser.add_argument(
        "--wandb-job-type",
        type=str,
        default=None,
        help="W&B job type override (default: Stormlog command name)",
    )
    parser.add_argument(
        "--wandb-log-artifacts",
        action="store_true",
        help="Upload Stormlog output bundles as W&B artifacts",
    )
    parser.add_argument(
        "--wandb-log-attribution",
        action="store_true",
        help="Log attribution HTML and top offenders to W&B when available",
    )


def ensure_wandb_available(config: WandbExportConfig) -> None:
    """Fail fast when the W&B feature is enabled without dependencies installed."""
    if config.enabled:
        _import_wandb()


def export_tracking_run_to_wandb(
    config: WandbExportConfig,
    *,
    command_name: str,
    session_summary: SessionSummary | None,
    stats: Mapping[str, Any],
    events: Sequence[Any],
    output_path: str | Path | None = None,
    telemetry_sink_dir: str | Path | None = None,
    oom_dump_path: str | Path | None = None,
) -> None:
    """Export one completed tracking session to W&B."""
    if not config.enabled:
        return

    wandb, run, managed = _resolve_run(
        config,
        command_name=command_name,
        session_summary=session_summary,
    )
    try:
        metrics = _tracking_metrics(stats)
        _update_summary(
            run,
            metrics
            | {
                "stormlog_chart_point_count": _tracking_chart_point_count(events),
            }
            | _session_summary_fields(session_summary)
            | _tracking_summary_fields(stats, output_path=output_path),
        )

        _log_tracking_time_series(run, events)

        if config.log_tables:
            _log_alerts_table(wandb, run, events)
            _log_tracking_visualizations(wandb, run, events)

        if config.log_artifacts:
            safe_session = _session_slug(session_summary)
            output_file = _coerce_existing_file(output_path)
            if output_file is not None:
                _log_file_artifact(
                    wandb,
                    run,
                    artifact_name=f"stormlog-track-output-{safe_session}",
                    artifact_type="stormlog-track-output",
                    path=output_file,
                )

            sink_dir = _coerce_existing_dir(telemetry_sink_dir)
            if sink_dir is not None:
                _log_directory_artifact(
                    wandb,
                    run,
                    artifact_name=f"stormlog-telemetry-sink-{safe_session}",
                    artifact_type="stormlog-telemetry-sink",
                    path=sink_dir,
                )

            oom_dir = _coerce_existing_dir(oom_dump_path)
            if oom_dir is not None:
                _log_directory_artifact(
                    wandb,
                    run,
                    artifact_name=f"stormlog-oom-dump-{safe_session}",
                    artifact_type="stormlog-oom-dump",
                    path=oom_dir,
                )

        if config.log_attribution:
            attribution_root = _coerce_existing_dir(oom_dump_path)
            if attribution_root is not None:
                _log_attribution_outputs(wandb, run, attribution_root)
    finally:
        if managed:
            run.finish()


def export_diagnose_bundle_to_wandb(
    config: WandbExportConfig,
    *,
    command_name: str,
    artifact_dir: str | Path,
) -> None:
    """Export one diagnose bundle directory to W&B."""
    if not config.enabled:
        return

    bundle_dir = _coerce_existing_dir(artifact_dir)
    if bundle_dir is None:
        raise FileNotFoundError(
            f"Diagnose artifact directory not found: {artifact_dir}"
        )

    manifest = _read_json_if_exists(bundle_dir / "manifest.json")
    diagnostic_summary = _read_json_if_exists(bundle_dir / "diagnostic_summary.json")
    session_summary = _session_summary_from_manifest(manifest)

    wandb, run, managed = _resolve_run(
        config,
        command_name=command_name,
        session_summary=session_summary,
    )
    try:
        metrics = _diagnose_metrics(diagnostic_summary, manifest)
        _update_summary(
            run,
            metrics
            | _session_summary_fields(session_summary)
            | _diagnose_summary_fields(bundle_dir, manifest),
        )

        if config.log_tables:
            _log_suggestions_table(wandb, run, diagnostic_summary)

        if config.log_artifacts:
            _log_directory_artifact(
                wandb,
                run,
                artifact_name=f"stormlog-diagnose-{_session_slug(session_summary)}",
                artifact_type="stormlog-diagnose",
                path=bundle_dir,
            )

        if config.log_attribution:
            _log_attribution_outputs(wandb, run, bundle_dir)
    finally:
        if managed:
            run.finish()


def _import_wandb() -> Any:
    try:
        import wandb  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        if exc.name == "wandb":
            raise ImportError(WANDB_INSTALL_GUIDANCE) from exc
        raise
    return wandb


def _resolve_run(
    config: WandbExportConfig,
    *,
    command_name: str,
    session_summary: SessionSummary | None,
) -> tuple[Any, Any, bool]:
    wandb = _import_wandb()
    active_run = getattr(wandb, "run", None)
    if active_run is not None:
        return wandb, active_run, False

    init_kwargs: dict[str, Any] = {
        "project": config.project or "stormlog",
        "entity": config.entity,
        "mode": config.mode,
        "name": config.run_name or _default_run_name(command_name, session_summary),
        "group": config.group or _default_group(session_summary),
        "job_type": config.job_type or command_name,
    }
    if config.run_id is not None:
        init_kwargs["id"] = config.run_id
        init_kwargs["resume"] = "allow"

    init_kwargs = {
        key: value for key, value in init_kwargs.items() if value is not None
    }
    run = wandb.init(**init_kwargs)
    return wandb, run, True


def _tracking_metrics(stats: Mapping[str, Any]) -> dict[str, Any]:
    metric_names = {
        "stormlog_peak_memory_bytes": "peak_memory",
        "stormlog_total_events": "total_events",
        "stormlog_alert_count": "alert_count",
        "stormlog_current_memory_allocated_bytes": "current_memory_allocated",
        "stormlog_current_memory_reserved_bytes": "current_memory_reserved",
        "stormlog_memory_utilization_percent": "memory_utilization_percent",
        "stormlog_total_allocations": "total_allocations",
        "stormlog_total_deallocations": "total_deallocations",
        "stormlog_total_allocation_bytes": "total_allocation_bytes",
        "stormlog_total_deallocation_bytes": "total_deallocation_bytes",
        "stormlog_tracking_duration_seconds": "tracking_duration_seconds",
        "stormlog_allocations_per_second": "allocations_per_second",
        "stormlog_bytes_allocated_per_second": "bytes_allocated_per_second",
        "stormlog_history_retained_events": "history_retained_events",
        "stormlog_history_dropped_events": "history_dropped_events",
        "stormlog_sink_rollover_count": "rollover_count",
        "stormlog_sink_pruned_segment_count": "pruned_segment_count",
        "stormlog_sink_pruned_bytes": "pruned_bytes",
        "stormlog_sink_retained_files": "final_retained_files",
        "stormlog_sink_retained_bytes": "final_retained_bytes",
    }
    metrics: dict[str, Any] = {}
    for wandb_key, stats_key in metric_names.items():
        value = stats.get(stats_key)
        if isinstance(value, (int, float, bool)) and not isinstance(value, complex):
            metrics[wandb_key] = value
    return metrics


def _diagnose_metrics(
    diagnostic_summary: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(diagnostic_summary, Mapping):
        return {}

    metrics: dict[str, Any] = {}
    for source_key, target_key in (
        ("allocated_bytes", "stormlog_allocated_bytes"),
        ("reserved_bytes", "stormlog_reserved_bytes"),
        ("peak_bytes", "stormlog_peak_bytes"),
        ("total_bytes", "stormlog_total_bytes"),
        ("utilization_ratio", "stormlog_utilization_ratio"),
        ("fragmentation_ratio", "stormlog_fragmentation_ratio"),
        ("num_ooms", "stormlog_num_ooms"),
    ):
        value = diagnostic_summary.get(source_key)
        if isinstance(value, (int, float, bool)) and not isinstance(value, complex):
            metrics[target_key] = value

    risk_flags = diagnostic_summary.get("risk_flags")
    if isinstance(risk_flags, Mapping):
        for key, value in risk_flags.items():
            if isinstance(value, bool):
                metrics[f"stormlog_risk_{key}"] = value

    if isinstance(manifest, Mapping):
        risk_detected = manifest.get("risk_detected")
        exit_code = manifest.get("exit_code")
        if isinstance(risk_detected, bool):
            metrics["stormlog_risk_detected"] = risk_detected
        if isinstance(exit_code, int):
            metrics["stormlog_exit_code"] = exit_code

    return metrics


def _tracking_summary_fields(
    stats: Mapping[str, Any],
    *,
    output_path: str | Path | None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for source_key, target_key in (
        ("backend", "stormlog_backend"),
        ("collector_health_status", "stormlog_collector_health_status"),
        ("collector_last_error", "stormlog_collector_last_error"),
        ("session_status", "stormlog_session_status"),
    ):
        value = stats.get(source_key)
        if value is not None:
            fields[target_key] = value

    output_file = _coerce_existing_file(output_path)
    if output_file is not None:
        fields["stormlog_output_file"] = output_file.name
    return fields


def _diagnose_summary_fields(
    artifact_dir: Path,
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "stormlog_artifact_dir": artifact_dir.name,
    }
    if isinstance(manifest, Mapping):
        for source_key, target_key in (
            ("risk_detected", "stormlog_risk_detected"),
            ("native_history_enabled", "stormlog_native_history_enabled"),
            ("session_status", "stormlog_session_status"),
        ):
            value = manifest.get(source_key)
            if value is not None:
                fields[target_key] = value
    return fields


def _session_summary_fields(summary: SessionSummary | None) -> dict[str, Any]:
    if summary is None:
        return {}
    fields: dict[str, Any] = {
        "stormlog_session_id": summary.session_id,
        "stormlog_session_source": summary.source,
        "stormlog_session_status": summary.status,
        "stormlog_rank": summary.rank,
        "stormlog_local_rank": summary.local_rank,
        "stormlog_world_size": summary.world_size,
    }
    if summary.job_id is not None:
        fields["stormlog_job_id"] = summary.job_id
    return fields


def _log_alerts_table(wandb: Any, run: Any, events: Sequence[Any]) -> None:
    rows: list[list[Any]] = []
    for event in events:
        event_type = _event_value(event, "event_type") or _event_value(event, "type")
        if event_type not in _ALERT_EVENT_TYPES:
            continue
        rows.append(
            [
                _event_timestamp_seconds(event),
                event_type,
                _event_value(event, "context"),
                _event_int_value(
                    event, "memory_allocated", "allocator_allocated_bytes"
                ),
                _event_int_value(event, "memory_reserved", "allocator_reserved_bytes"),
                _event_int_value(event, "memory_change", "allocator_change_bytes"),
                _event_value(event, "job_id"),
                _event_value(event, "rank"),
            ]
        )
    if not rows:
        return
    run.log(
        {
            "stormlog_alerts": wandb.Table(
                columns=[
                    "timestamp_s",
                    "event_type",
                    "context",
                    "memory_allocated_bytes",
                    "memory_reserved_bytes",
                    "memory_change_bytes",
                    "job_id",
                    "rank",
                ],
                data=rows[-250:],
            )
        }
    )


def _log_suggestions_table(
    wandb: Any,
    run: Any,
    diagnostic_summary: Mapping[str, Any] | None,
) -> None:
    if not isinstance(diagnostic_summary, Mapping):
        return
    suggestions = diagnostic_summary.get("suggestions")
    if not isinstance(suggestions, list) or not suggestions:
        return
    rows = [
        [index + 1, str(suggestion)] for index, suggestion in enumerate(suggestions)
    ]
    run.log(
        {
            "stormlog_diagnostic_suggestions": wandb.Table(
                columns=["index", "suggestion"],
                data=rows,
            )
        }
    )


def _log_tracking_time_series(run: Any, events: Sequence[Any]) -> None:
    rows = _tracking_timeline_rows(events)
    for row in rows:
        payload = {
            "stormlog_timeline_elapsed_seconds": row["elapsed_seconds"],
            "stormlog_timeline_allocated_bytes": row["allocated_bytes"],
            "stormlog_timeline_reserved_bytes": row["reserved_bytes"],
            "stormlog_timeline_change_bytes": row["change_bytes"],
            "stormlog_timeline_device_used_bytes": row["device_used_bytes"],
            "stormlog_timeline_utilization_percent": row["utilization_percent"],
        }
        filtered_payload = {
            key: value for key, value in payload.items() if value is not None
        }
        if filtered_payload:
            run.log(filtered_payload)


def _log_tracking_visualizations(wandb: Any, run: Any, events: Sequence[Any]) -> None:
    rows = _tracking_timeline_rows(events)
    if not rows:
        return

    run.log(
        {
            "stormlog_memory_timeline_table": wandb.Table(
                columns=[
                    "sample_index",
                    "elapsed_seconds",
                    "event_type",
                    "memory_allocated_bytes",
                    "memory_reserved_bytes",
                    "memory_change_bytes",
                    "device_used_bytes",
                    "utilization_percent",
                    "context",
                    "rank",
                ],
                data=[
                    [
                        row["sample_index"],
                        row["elapsed_seconds"],
                        row["event_type"],
                        row["allocated_bytes"],
                        row["reserved_bytes"],
                        row["change_bytes"],
                        row["device_used_bytes"],
                        row["utilization_percent"],
                        row["context"],
                        row["rank"],
                    ]
                    for row in rows
                ],
            )
        }
    )

    plot_api = getattr(wandb, "plot", None)
    line_series = getattr(plot_api, "line_series", None)
    if callable(line_series):
        elapsed = [float(row["elapsed_seconds"]) for row in rows]
        run.log(
            {
                "stormlog_memory_timeline_plot": line_series(
                    xs=elapsed,
                    ys=[
                        [int(row["allocated_bytes"] or 0) for row in rows],
                        [int(row["reserved_bytes"] or 0) for row in rows],
                        [int(row["device_used_bytes"] or 0) for row in rows],
                    ],
                    keys=["allocated_bytes", "reserved_bytes", "device_used_bytes"],
                    title="Stormlog Memory Timeline",
                    xname="Elapsed Seconds",
                )
            }
        )

        if any(row["utilization_percent"] is not None for row in rows):
            run.log(
                {
                    "stormlog_memory_utilization_plot": line_series(
                        xs=elapsed,
                        ys=[[float(row["utilization_percent"] or 0.0) for row in rows]],
                        keys=["utilization_percent"],
                        title="Stormlog Memory Utilization",
                        xname="Elapsed Seconds",
                    )
                }
            )

    run.log({"stormlog_tracking_dashboard": wandb.Html(_tracking_dashboard_html(rows))})


def _log_attribution_outputs(wandb: Any, run: Any, root: Path) -> None:
    html_path = root / TRACE_HTML_ANNOTATED_FILENAME
    if html_path.exists():
        run.log(
            {
                "stormlog_attribution_html": wandb.Html(
                    html_path.read_text(encoding="utf-8")
                )
            }
        )

    tensor_rows = _tensor_attribution_rows(root / TENSOR_ATTRIBUTION_FILENAME)
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


def _tensor_attribution_rows(path: Path) -> list[list[Any]]:
    payload = _read_json_if_exists(path)
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


def _log_file_artifact(
    wandb: Any,
    run: Any,
    *,
    artifact_name: str,
    artifact_type: str,
    path: Path,
) -> None:
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(local_path=str(path), name=path.name)
    run.log_artifact(artifact)


def _log_directory_artifact(
    wandb: Any,
    run: Any,
    *,
    artifact_name: str,
    artifact_type: str,
    path: Path,
) -> None:
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_dir(local_path=str(path), name=path.name)
    run.log_artifact(artifact)


def _tracking_chart_point_count(events: Sequence[Any]) -> int:
    return len(_tracking_timeline_rows(events))


def _tracking_timeline_rows(events: Sequence[Any]) -> list[dict[str, Any]]:
    timeline_rows: list[dict[str, Any]] = []
    first_timestamp: float | None = None

    for event in events:
        timestamp_s = _event_timestamp_seconds(event)
        if timestamp_s is None:
            continue
        if first_timestamp is None:
            first_timestamp = timestamp_s

        allocated = _event_int_value(
            event, "memory_allocated", "allocator_allocated_bytes"
        )
        reserved = _event_int_value(
            event, "memory_reserved", "allocator_reserved_bytes"
        )
        change = _event_int_value(event, "memory_change", "allocator_change_bytes")
        device_used = _event_int_value(event, "device_used", "device_used_bytes")
        device_total = _event_int_value(event, "device_total", "device_total_bytes")

        if device_used is None:
            candidates = [value for value in (allocated, reserved) if value is not None]
            device_used = max(candidates) if candidates else None

        utilization_percent: float | None = None
        if (
            isinstance(device_used, int)
            and isinstance(device_total, int)
            and device_total > 0
        ):
            utilization_percent = (float(device_used) / float(device_total)) * 100.0

        timeline_rows.append(
            {
                "sample_index": len(timeline_rows),
                "elapsed_seconds": timestamp_s - first_timestamp,
                "event_type": str(
                    _event_value(event, "event_type")
                    or _event_value(event, "type")
                    or "sample"
                ),
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
                "change_bytes": change,
                "device_used_bytes": device_used,
                "utilization_percent": utilization_percent,
                "context": _event_value(event, "context"),
                "rank": _event_value(event, "rank"),
            }
        )

    return _sample_timeline_rows(timeline_rows)


def _sample_timeline_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) <= _TIMELINE_MAX_POINTS:
        return list(rows)

    stride = int(math.ceil(len(rows) / _TIMELINE_MAX_POINTS))
    sampled = list(rows[::stride])
    last_row = rows[-1]
    if not sampled or sampled[-1]["sample_index"] != last_row["sample_index"]:
        sampled.append(last_row)
    return sampled


def _tracking_dashboard_html(rows: Sequence[Mapping[str, Any]]) -> str:
    allocated_series = [int(row.get("allocated_bytes") or 0) for row in rows]
    reserved_series = [int(row.get("reserved_bytes") or 0) for row in rows]
    utilization_values = [
        float(value)
        for value in (row.get("utilization_percent") for row in rows)
        if isinstance(value, (int, float))
    ]
    alert_rows = [
        row for row in rows if str(row.get("event_type", "")) in _ALERT_EVENT_TYPES
    ][-8:]

    chart_min = 0.0
    chart_max = float(
        max(allocated_series + reserved_series)
        if allocated_series or reserved_series
        else 1
    )
    allocated_points = _svg_polyline_points(
        [float(value) for value in allocated_series],
        width=_DASHBOARD_WIDTH,
        height=_DASHBOARD_HEIGHT,
        min_value=chart_min,
        max_value=chart_max,
    )
    reserved_points = _svg_polyline_points(
        [float(value) for value in reserved_series],
        width=_DASHBOARD_WIDTH,
        height=_DASHBOARD_HEIGHT,
        min_value=chart_min,
        max_value=chart_max,
    )

    cards = [
        ("samples", str(len(rows))),
        (
            "peak allocated",
            _format_bytes(max(allocated_series) if allocated_series else 0),
        ),
        (
            "peak reserved",
            _format_bytes(max(reserved_series) if reserved_series else 0),
        ),
        (
            "max utilization",
            f"{max(utilization_values):.1f}%" if utilization_values else "n/a",
        ),
    ]
    card_html = "".join(
        f"<div class='card'><div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div></div>"
        for label, value in cards
    )
    alerts_html = "".join(
        "<tr>"
        f"<td>{row.get('sample_index')}</td>"
        f"<td>{html.escape(str(row.get('event_type', '')))}</td>"
        f"<td>{row.get('elapsed_seconds', 0.0):.2f}</td>"
        f"<td>{html.escape(str(row.get('context') or ''))}</td>"
        "</tr>"
        for row in alert_rows
    )
    alerts_body = (
        alerts_html or "<tr><td colspan='4'>No alert events captured.</td></tr>"
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:18px;color:#1f2937;background:#fff;}"
        ".cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:18px;}"
        ".card{border:1px solid #dbe3ea;border-radius:12px;padding:12px;background:#f8fafc;}"
        ".label{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;}"
        ".value{font-size:22px;font-weight:700;margin-top:6px;}"
        ".legend{display:flex;gap:18px;margin:10px 0 14px;font-size:13px;color:#475569;}"
        ".swatch{display:inline-block;width:10px;height:10px;border-radius:999px;margin-right:6px;}"
        "table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px;}"
        "th,td{padding:8px 10px;border-bottom:1px solid #e2e8f0;text-align:left;}"
        "th{color:#475569;font-weight:600;background:#f8fafc;}"
        "h2{margin:0 0 12px;font-size:20px;}"
        "p{margin:0 0 10px;color:#475569;}"
        "</style></head><body>"
        "<h2>Stormlog Tracking Dashboard</h2>"
        "<p>Sampled timeline exported to Weights & Biases from Stormlog tracking events.</p>"
        f"<div class='cards'>{card_html}</div>"
        "<svg viewBox='0 0 720 260' width='100%' role='img' aria-label='Stormlog memory timeline'>"
        "<rect x='0' y='0' width='720' height='260' fill='#ffffff' stroke='#e2e8f0' rx='12'/>"
        f"<polyline fill='none' stroke='#2563eb' stroke-width='3' points='{allocated_points}'/>"
        f"<polyline fill='none' stroke='#f97316' stroke-width='3' points='{reserved_points}'/>"
        "</svg>"
        "<div class='legend'>"
        "<span><span class='swatch' style='background:#2563eb;'></span>Allocated</span>"
        "<span><span class='swatch' style='background:#f97316;'></span>Reserved</span>"
        "</div>"
        "<table><thead><tr><th>sample</th><th>event</th><th>elapsed (s)</th><th>context</th></tr></thead>"
        f"<tbody>{alerts_body}</tbody></table>"
        "</body></html>"
    )


def _svg_polyline_points(
    values: Sequence[float],
    *,
    width: int,
    height: int,
    min_value: float,
    max_value: float,
) -> str:
    if not values:
        return ""
    inner_width = float(width - (_DASHBOARD_PADDING * 2))
    inner_height = float(height - (_DASHBOARD_PADDING * 2))
    span = max(max_value - min_value, 1.0)
    point_count = max(len(values) - 1, 1)
    points: list[str] = []
    for index, value in enumerate(values):
        x = _DASHBOARD_PADDING + (float(index) / float(point_count)) * inner_width
        normalized = (float(value) - min_value) / span
        y = height - _DASHBOARD_PADDING - (normalized * inner_height)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    units = ["KB", "MB", "GB", "TB", "PB"]
    scaled = float(value)
    unit = "B"
    for unit in units:
        scaled /= 1024.0
        if scaled < 1024.0:
            return f"{scaled:.2f} {unit}"
    return f"{scaled:.2f} {unit}"


def _update_summary(run: Any, payload: Mapping[str, Any]) -> None:
    if not payload:
        return
    summary = getattr(run, "summary", None)
    if summary is None:
        return
    if hasattr(summary, "update"):
        summary.update(payload)
        return
    for key, value in payload.items():
        summary[key] = value


def _session_summary_from_manifest(
    manifest: Mapping[str, Any] | None,
) -> SessionSummary | None:
    if not isinstance(manifest, Mapping):
        return None
    session_payload = manifest.get("session")
    if not isinstance(session_payload, Mapping):
        return None
    try:
        return session_summary_from_dict(session_payload)
    except ValueError:
        return None


def _default_run_name(
    command_name: str,
    session_summary: SessionSummary | None,
) -> str:
    if session_summary is None:
        return command_name
    return f"{command_name}-{session_summary.session_id[:8]}"


def _default_group(summary: SessionSummary | None) -> str | None:
    if summary is None or summary.job_id is None:
        return None
    return summary.job_id


def _session_slug(summary: SessionSummary | None) -> str:
    raw = summary.session_id if summary is not None else "session"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw)
    return slug or "session"


def _event_value(event: Any, name: str) -> Any:
    if isinstance(event, Mapping):
        return event.get(name)
    return getattr(event, name, None)


def _event_int_value(event: Any, *names: str) -> int | None:
    for name in names:
        value = _event_value(event, name)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return None


def _event_timestamp_seconds(event: Any) -> float | None:
    value = _event_value(event, "timestamp")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    value_ns = _event_value(event, "timestamp_ns")
    if isinstance(value_ns, int) and not isinstance(value_ns, bool):
        return float(value_ns) / 1_000_000_000.0
    return None


def _normalized_optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _coerce_existing_file(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.exists() and path.is_file():
        return path
    return None


def _coerce_existing_dir(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.exists() and path.is_dir():
        return path
    return None


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


__all__ = [
    "WANDB_INSTALL_GUIDANCE",
    "WandbExportConfig",
    "add_wandb_arguments",
    "ensure_wandb_available",
    "export_diagnose_bundle_to_wandb",
    "export_tracking_run_to_wandb",
    "wandb_config_from_namespace",
]
