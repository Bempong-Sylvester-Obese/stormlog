"""Structured phase tracking demo using CPUMemoryTracker."""

from __future__ import annotations

import json
import time
from pathlib import Path

from stormlog import CPUMemoryTracker
from stormlog.phases import parse_phase_boundary


def _do_work(units: int) -> int:
    total = 0
    for value in range(units):
        total += value * value
    return total


def main() -> None:
    tracker = CPUMemoryTracker(sampling_interval=0.05)
    tracker.start_tracking()

    with tracker.phase("train", metadata={"epoch": 1}):
        with tracker.phase("forward", metadata={"microbatch": 4}):
            _do_work(100_000)
        with tracker.phase("backward", metadata={"microbatch": 4}):
            _do_work(75_000)

    with tracker.phase("evaluate", metadata={"split": "validation"}):
        time.sleep(0.05)
        _do_work(50_000)

    tracker.stop_tracking()

    output_dir = Path("artifacts/phase_tracking_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase_events.json"
    tracker.export_events(str(output_path), format="json")
    records = json.loads(output_path.read_text(encoding="utf-8"))

    phase_records = [
        record for record in records if str(record.get("event_type", "")).startswith("phase_")
    ]
    print(f"Wrote {len(records)} telemetry records to {output_path}")
    print(f"Structured phase boundaries: {len(phase_records)}")
    for record in phase_records:
        scope = parse_phase_boundary(record)
        if scope is None:
            continue
        path = " / ".join(scope.path)
        print(f"- {record['event_type']}: {path} attrs={scope.attributes}")


if __name__ == "__main__":
    main()
