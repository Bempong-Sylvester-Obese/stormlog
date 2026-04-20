"""Core helpers shared by Stormlog's optional W&B exporters."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..session import SessionSummary, session_summary_from_dict

WANDB_INSTALL_GUIDANCE = (
    "Weights & Biases integration requires optional dependencies. "
    "Install with `pip install 'stormlog[wandb]'`."
)


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
        import_wandb()


def import_wandb() -> Any:
    try:
        import wandb
    except ModuleNotFoundError as exc:
        if exc.name == "wandb":
            raise ImportError(WANDB_INSTALL_GUIDANCE) from exc
        raise
    return wandb


def resolve_run(
    config: WandbExportConfig,
    *,
    command_name: str,
    session_summary: SessionSummary | None,
) -> tuple[Any, Any, bool]:
    wandb = import_wandb()
    active_run = getattr(wandb, "run", None)
    if active_run is not None:
        return wandb, active_run, False

    init_kwargs: dict[str, Any] = {
        "project": config.project or "stormlog",
        "entity": config.entity,
        "mode": config.mode,
        "name": config.run_name or _default_run_name(command_name, session_summary),
        "group": config.group or default_group(session_summary),
        "job_type": config.job_type or command_name,
    }
    if config.run_id is not None:
        init_kwargs["id"] = config.run_id
        init_kwargs["resume"] = "allow"

    run = wandb.init(
        **{key: value for key, value in init_kwargs.items() if value is not None}
    )
    return wandb, run, True


def update_summary(run: Any, payload: Mapping[str, Any]) -> None:
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


def session_summary_from_manifest(
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


def session_summary_fields(summary: SessionSummary | None) -> dict[str, Any]:
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


def default_group(summary: SessionSummary | None) -> str | None:
    if summary is None or summary.job_id is None:
        return None
    return summary.job_id


def session_slug(summary: SessionSummary | None) -> str:
    raw = summary.session_id if summary is not None else "session"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw)
    return slug or "session"


def coerce_existing_file(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.exists() and path.is_file():
        return path
    return None


def coerce_existing_dir(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.exists() and path.is_dir():
        return path
    return None


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def log_file_artifact(
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


def log_directory_artifact(
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


def materialize_html_file(
    *,
    html_text: str,
    file_name: str,
    output_root: Path | None,
) -> Path:
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        target_path = output_root / file_name
        target_path.write_text(html_text, encoding="utf-8")
        return target_path

    temp_dir = Path(tempfile.mkdtemp(prefix="stormlog-wandb-"))
    target_path = temp_dir / file_name
    target_path.write_text(html_text, encoding="utf-8")
    target_path.chmod(0o600)
    return target_path


def _default_run_name(
    command_name: str,
    session_summary: SessionSummary | None,
) -> str:
    if session_summary is None:
        return command_name
    return f"{command_name}-{session_summary.session_id[:8]}"


def _normalized_optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
