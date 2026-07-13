"""Stable public facade for run envelope and attachment catalog helpers."""

from . import _run_catalog_models as _models
from ._run_catalog_artifacts import (
    diagnose_attachment_rows,
    envelope_attachment_rows,
    flat_telemetry_attachment_rows,
    local_attachment_row,
    oom_attachment_rows,
    rollup_attachment_rows,
    sidecar_attachment_rows,
    sink_attachment_rows,
    sink_segment_attachment_rows,
)
from ._run_catalog_context import (
    build_identity_index,
    build_run_contexts,
    run_attachment_matches,
    run_id_for_identity,
    run_matches,
)
from ._run_catalog_models import (
    RUN_ENVELOPE_FILENAME,
    RUN_ENVELOPE_FORMAT,
    RUN_ENVELOPE_SCHEMA_VERSION,
    AttachmentStorage,
    CatalogRunAttachment,
    CatalogRunSessionRef,
    RunContext,
    RunIdentityConflict,
    RunIdentityIndex,
)
from ._run_catalog_parser import (
    attachment_storage_or_default,
    attachment_storage_or_none,
    is_run_envelope,
    run_envelope_from_payload,
)

CatalogRunEnvelope = _models.CatalogRunEnvelope
RunAttachmentFilter = _models.RunAttachmentFilter
RunAttachmentRow = _models.RunAttachmentRow
RunFilter = _models.RunFilter
RunRow = _models.RunRow

__all__ = [
    "AttachmentStorage",
    "CatalogRunAttachment",
    "CatalogRunSessionRef",
    "RUN_ENVELOPE_FILENAME",
    "RUN_ENVELOPE_FORMAT",
    "RUN_ENVELOPE_SCHEMA_VERSION",
    "RunContext",
    "RunIdentityConflict",
    "RunIdentityIndex",
    "attachment_storage_or_default",
    "attachment_storage_or_none",
    "build_identity_index",
    "build_run_contexts",
    "diagnose_attachment_rows",
    "envelope_attachment_rows",
    "flat_telemetry_attachment_rows",
    "is_run_envelope",
    "local_attachment_row",
    "oom_attachment_rows",
    "rollup_attachment_rows",
    "run_attachment_matches",
    "run_envelope_from_payload",
    "run_id_for_identity",
    "run_matches",
    "sidecar_attachment_rows",
    "sink_attachment_rows",
    "sink_segment_attachment_rows",
]
