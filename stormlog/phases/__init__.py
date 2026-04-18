"""Structured phase telemetry helpers for trackers and analysis."""

from .policy import (
    PhaseAttribution,
    PhaseSummary,
    attribute_active_spans,
    format_phase_path,
    merge_phase_attributions,
    phase_attribution_to_payload,
    resolve_phase_for_event,
    summarize_phase_attribution,
    summarize_phase_resolution,
)
from .replay import (
    PhaseBoundaryRecord,
    PhaseReplayIndex,
    PhaseSpan,
    is_phase_boundary_event,
    parse_phase_boundary,
)
from .runtime import (
    PHASE_ENTER_EVENT,
    PHASE_EXIT_EVENT,
    PHASE_SCOPE_ATTRIBUTES_KEY,
    PHASE_SCOPE_METADATA_KEY,
    PhaseBoundary,
    PhaseHandle,
    PhaseProtocolError,
    PhaseRecorder,
    PhaseToken,
)

__all__ = [
    "PHASE_ENTER_EVENT",
    "PHASE_EXIT_EVENT",
    "PHASE_SCOPE_METADATA_KEY",
    "PHASE_SCOPE_ATTRIBUTES_KEY",
    "PhaseAttribution",
    "PhaseSummary",
    "attribute_active_spans",
    "PhaseBoundary",
    "PhaseBoundaryRecord",
    "PhaseHandle",
    "PhaseProtocolError",
    "PhaseRecorder",
    "PhaseReplayIndex",
    "PhaseSpan",
    "PhaseToken",
    "phase_attribution_to_payload",
    "resolve_phase_for_event",
    "format_phase_path",
    "is_phase_boundary_event",
    "merge_phase_attributions",
    "parse_phase_boundary",
    "summarize_phase_attribution",
    "summarize_phase_resolution",
]
