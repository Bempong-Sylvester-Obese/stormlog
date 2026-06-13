"""Inference profiling helpers for OpenAI-compatible serving endpoints."""

__all__ = ["InferenceProfiler", "analyze_inference_events", "run_profile"]


def __getattr__(name: str) -> object:
    if name == "analyze_inference_events":
        from .analysis import analyze_inference_events

        return analyze_inference_events
    if name in {"InferenceProfiler", "run_profile"}:
        from .profile import InferenceProfiler, run_profile

        return {"InferenceProfiler": InferenceProfiler, "run_profile": run_profile}[
            name
        ]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
