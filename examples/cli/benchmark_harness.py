"""Benchmark harness for profiling overhead and artifact-size budget checks."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from stormlog.cpu_profiler import CPUMemoryTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_VERSION = "v0.3"
DEFAULT_BUDGETS_PATH = REPO_ROOT / "docs" / "benchmarks" / "v0.2_budgets.json"
DEFAULT_BASELINE_PATH = REPO_ROOT / "docs" / "benchmarks" / "v0.3_baseline.json"
DEFAULT_TOLERANCES_PATH = (
    REPO_ROOT / "docs" / "benchmarks" / "v0.3_tolerances.json"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "latest.json"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "benchmarks" / "scenarios"
DEFAULT_GATE_MODE = "budget"
_BASELINE_CONFIG_KEYS = (
    "iterations",
    "allocation_kb",
    "default_interval",
    "lowfreq_interval",
)

_BUDGET_KEY_BY_METRIC: Dict[str, str] = {
    "runtime_overhead_pct": "runtime_overhead_pct_max",
    "cpu_overhead_pct": "cpu_overhead_pct_max",
    "sampling_impact_pct": "sampling_impact_pct_max",
    "artifact_growth_bytes": "artifact_growth_bytes_max",
}
_METRIC_KEYS = tuple(_BUDGET_KEY_BY_METRIC.keys())


@dataclass
class ScenarioResult:
    """Single benchmark scenario output."""

    name: str
    wall_seconds: float
    cpu_seconds: float
    checksum: int
    event_count: int
    peak_memory_bytes: int
    artifact_size_bytes: int
    artifact_dir: str


def _run_workload(iterations: int, allocation_kb: int) -> int:
    """Run a deterministic CPU workload with allocation churn."""
    checksum = 0
    block_bytes = max(1, allocation_kb) * 1024

    for step in range(max(1, iterations)):
        payload = bytearray(block_bytes)
        marker = (step * 17) % 251
        for offset in range(0, len(payload), 4096):
            payload[offset] = marker
            checksum += payload[offset]
        checksum ^= len(payload)

    return checksum


def _directory_size_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _run_scenario(
    name: str,
    *,
    iterations: int,
    allocation_kb: int,
    artifact_root: Path,
    sampling_interval: Optional[float],
) -> ScenarioResult:
    scenario_dir = artifact_root / name
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    tracker: Optional[CPUMemoryTracker] = None
    if sampling_interval is not None:
        tracker = CPUMemoryTracker(
            sampling_interval=sampling_interval,
            enable_alerts=False,
        )
        tracker.start_tracking()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    checksum = _run_workload(iterations=iterations, allocation_kb=allocation_kb)
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start

    event_count = 0
    peak_memory_bytes = 0

    if tracker is not None:
        tracker.stop_tracking()
        events = tracker.get_events()
        stats = tracker.get_statistics()
        event_count = len(events)
        peak_memory_bytes = int(stats.get("peak_memory", 0))
        tracker.export_events(str(scenario_dir / "events.json"), format="json")

    summary_path = scenario_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "name": name,
                "sampling_interval": sampling_interval,
                "iterations": iterations,
                "allocation_kb": allocation_kb,
                "wall_seconds": wall_seconds,
                "cpu_seconds": cpu_seconds,
                "checksum": checksum,
                "event_count": event_count,
                "peak_memory_bytes": peak_memory_bytes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return ScenarioResult(
        name=name,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        checksum=checksum,
        event_count=event_count,
        peak_memory_bytes=peak_memory_bytes,
        artifact_size_bytes=_directory_size_bytes(scenario_dir),
        artifact_dir=str(scenario_dir),
    )


def _pct_overhead(baseline: float, measured: float) -> float:
    if baseline <= 0:
        return 0.0 if measured <= 0 else float("inf")
    return max(0.0, ((measured - baseline) / baseline) * 100.0)


def _metric_values_from_mapping(
    raw_values: Mapping[str, Any],
    *,
    label: str,
) -> Dict[str, float]:
    missing_keys = [key for key in _METRIC_KEYS if key not in raw_values]
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"{label} missing metric keys: {missing}")
    return {key: float(raw_values[key]) for key in _METRIC_KEYS}


def _normalize_comparison_config(
    raw_config: Mapping[str, Any],
    *,
    label: str,
) -> Dict[str, float]:
    missing_keys = [key for key in _BASELINE_CONFIG_KEYS if key not in raw_config]
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"{label} missing config keys: {missing}")
    return {
        "iterations": int(raw_config["iterations"]),
        "allocation_kb": int(raw_config["allocation_kb"]),
        "default_interval": float(raw_config["default_interval"]),
        "lowfreq_interval": float(raw_config["lowfreq_interval"]),
    }


def load_budget_thresholds(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    budgets_obj = payload.get("budgets", payload)

    missing_keys = [
        key for key in _BUDGET_KEY_BY_METRIC.values() if key not in budgets_obj
    ]
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Budget file missing keys: {missing}")

    return {key: float(value) for key, value in budgets_obj.items()}


def load_regression_baseline(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != REPORT_VERSION:
        raise ValueError(
            f"Baseline file version must be {REPORT_VERSION}, "
            f"found {payload.get('version')!r}"
        )

    raw_config = payload.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Baseline file missing config mapping")

    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise ValueError("Baseline file missing metrics mapping")

    return {
        "version": REPORT_VERSION,
        "config": _normalize_comparison_config(raw_config, label="Baseline file"),
        "metrics": _metric_values_from_mapping(raw_metrics, label="Baseline file"),
    }


def load_regression_tolerances(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != REPORT_VERSION:
        raise ValueError(
            f"Tolerance file version must be {REPORT_VERSION}, "
            f"found {payload.get('version')!r}"
        )

    tolerances_obj = payload.get("tolerances", payload)
    if not isinstance(tolerances_obj, Mapping):
        raise ValueError("Tolerance file missing tolerances mapping")

    return _metric_values_from_mapping(tolerances_obj, label="Tolerance file")


def evaluate_budgets(
    metrics: Dict[str, float],
    budgets: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}
    for metric_key, budget_key in _BUDGET_KEY_BY_METRIC.items():
        value = float(metrics[metric_key])
        max_allowed = float(budgets[budget_key])
        checks[metric_key] = {
            "value": value,
            "max_allowed": max_allowed,
            "passed": value <= max_allowed,
        }
    return checks


def validate_regression_config(
    current_config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
) -> None:
    normalized_current = _normalize_comparison_config(
        current_config, label="Current benchmark config"
    )
    normalized_baseline = _normalize_comparison_config(
        baseline_config, label="Baseline file"
    )

    for key in _BASELINE_CONFIG_KEYS:
        if normalized_current[key] != normalized_baseline[key]:
            raise ValueError(
                "Baseline config mismatch for "
                f"{key}: current={normalized_current[key]!r}, "
                f"baseline={normalized_baseline[key]!r}"
            )


def evaluate_regressions(
    metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    tolerances: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}
    for metric_key in _METRIC_KEYS:
        current_value = float(metrics[metric_key])
        baseline_value = float(baseline_metrics[metric_key])
        max_regression = float(tolerances[metric_key])
        delta = current_value - baseline_value
        checks[metric_key] = {
            "current_value": current_value,
            "baseline_value": baseline_value,
            "delta": delta,
            "max_regression": max_regression,
            "passed": delta <= max_regression,
        }
    return checks


def _format_metric_value(metric_key: str, value: float) -> str:
    if metric_key == "artifact_growth_bytes":
        return f"{value:.0f}"
    return f"{value:.2f}"


def format_regression_summary(report: Mapping[str, Any]) -> list[str]:
    checks = report.get("regression_checks", {})
    if not isinstance(checks, Mapping):
        return []

    lines: list[str] = []
    for metric_key in _METRIC_KEYS:
        raw_check = checks.get(metric_key)
        if not isinstance(raw_check, Mapping):
            continue
        current_value = float(raw_check["current_value"])
        baseline_value = float(raw_check["baseline_value"])
        delta = float(raw_check["delta"])
        max_regression = float(raw_check["max_regression"])
        status = "PASS" if bool(raw_check["passed"]) else "FAIL"
        lines.append(
            f"{metric_key}: current={_format_metric_value(metric_key, current_value)} "
            f"baseline={_format_metric_value(metric_key, baseline_value)} "
            f"delta={_format_metric_value(metric_key, delta)} "
            f"allowed={_format_metric_value(metric_key, max_regression)} "
            f"[{status}]"
        )
    return lines


def run_benchmark_harness(
    *,
    iterations: int,
    allocation_kb: int,
    default_interval: float,
    lowfreq_interval: float,
    gate_mode: str,
    budgets_path: Optional[Path],
    baseline_path: Optional[Path],
    tolerances_path: Optional[Path],
    artifact_root: Path,
    output_path: Path,
) -> Dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "unprofiled": _run_scenario(
            "unprofiled",
            iterations=iterations,
            allocation_kb=allocation_kb,
            artifact_root=artifact_root,
            sampling_interval=None,
        ),
        "tracked_default": _run_scenario(
            "tracked_default",
            iterations=iterations,
            allocation_kb=allocation_kb,
            artifact_root=artifact_root,
            sampling_interval=default_interval,
        ),
        "tracked_lowfreq": _run_scenario(
            "tracked_lowfreq",
            iterations=iterations,
            allocation_kb=allocation_kb,
            artifact_root=artifact_root,
            sampling_interval=lowfreq_interval,
        ),
    }

    metrics = {
        "runtime_overhead_pct": _pct_overhead(
            scenarios["unprofiled"].wall_seconds,
            scenarios["tracked_default"].wall_seconds,
        ),
        "cpu_overhead_pct": _pct_overhead(
            scenarios["unprofiled"].cpu_seconds,
            scenarios["tracked_default"].cpu_seconds,
        ),
        "sampling_impact_pct": _pct_overhead(
            scenarios["tracked_lowfreq"].wall_seconds,
            scenarios["tracked_default"].wall_seconds,
        ),
        "artifact_growth_bytes": float(
            max(
                0,
                scenarios["tracked_default"].artifact_size_bytes
                - scenarios["unprofiled"].artifact_size_bytes,
            )
        ),
    }

    config = {
        "iterations": iterations,
        "allocation_kb": allocation_kb,
        "default_interval": default_interval,
        "lowfreq_interval": lowfreq_interval,
        "artifact_root": str(artifact_root),
    }

    report: Dict[str, Any] = {
        "version": REPORT_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate_mode": gate_mode,
        "config": config,
        "scenarios": {name: asdict(result) for name, result in scenarios.items()},
        "metrics": metrics,
    }

    if gate_mode == "budget":
        if budgets_path is None:
            raise ValueError("Budget gate mode requires a budgets path")
        budgets = load_budget_thresholds(budgets_path)
        budget_checks = evaluate_budgets(metrics, budgets)
        report["config"]["budgets_path"] = str(budgets_path)
        report["budgets"] = budgets
        report["budget_checks"] = budget_checks
        report["passed"] = all(
            bool(check["passed"]) for check in budget_checks.values()
        )
    elif gate_mode == "regression":
        if baseline_path is None or tolerances_path is None:
            raise ValueError(
                "Regression gate mode requires baseline and tolerance paths"
            )
        baseline = load_regression_baseline(baseline_path)
        validate_regression_config(config, baseline["config"])
        tolerances = load_regression_tolerances(tolerances_path)
        regression_checks = evaluate_regressions(
            metrics,
            baseline["metrics"],
            tolerances,
        )
        report["config"]["baseline_path"] = str(baseline_path)
        report["config"]["tolerances_path"] = str(tolerances_path)
        report["baseline"] = baseline
        report["tolerances"] = tolerances
        report["regression_checks"] = regression_checks
        report["passed"] = all(
            bool(check["passed"]) for check in regression_checks.values()
        )
    else:
        raise ValueError(f"Unsupported gate mode: {gate_mode}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark profiler overhead and enforce benchmark gates.",
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--allocation-kb", type=int, default=512)
    parser.add_argument("--default-interval", type=float, default=0.1)
    parser.add_argument("--lowfreq-interval", type=float, default=0.5)
    parser.add_argument(
        "--gate-mode",
        choices=["budget", "regression"],
        default=DEFAULT_GATE_MODE,
        help="Gate mode to evaluate when --check is used.",
    )
    parser.add_argument("--budgets", type=Path, default=DEFAULT_BUDGETS_PATH)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--tolerances", type=Path, default=DEFAULT_TOLERANCES_PATH)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return a non-zero exit code when any budget is violated.",
    )
    args = parser.parse_args(argv)

    report = run_benchmark_harness(
        iterations=args.iterations,
        allocation_kb=args.allocation_kb,
        default_interval=args.default_interval,
        lowfreq_interval=args.lowfreq_interval,
        gate_mode=args.gate_mode,
        budgets_path=args.budgets if args.gate_mode == "budget" else None,
        baseline_path=args.baseline if args.gate_mode == "regression" else None,
        tolerances_path=(
            args.tolerances if args.gate_mode == "regression" else None
        ),
        artifact_root=args.artifact_root,
        output_path=args.output,
    )

    print(f"Benchmark report written to: {args.output}")
    if args.gate_mode == "budget":
        print(f"Runtime overhead: {report['metrics']['runtime_overhead_pct']:.2f}%")
        print(f"CPU overhead: {report['metrics']['cpu_overhead_pct']:.2f}%")
        print(f"Sampling impact: {report['metrics']['sampling_impact_pct']:.2f}%")
        print(f"Artifact growth: {report['metrics']['artifact_growth_bytes']:.0f} bytes")
        print(f"Budget status: {'PASS' if report['passed'] else 'FAIL'}")
    else:
        print(f"Regression baseline: {args.baseline}")
        for line in format_regression_summary(report):
            print(line)
        print(f"Regression status: {'PASS' if report['passed'] else 'FAIL'}")

    if args.check and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
