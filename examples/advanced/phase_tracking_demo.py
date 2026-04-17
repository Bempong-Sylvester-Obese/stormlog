"""Phase-aware tracking demo using nested tracker scopes."""

from __future__ import annotations

import time
from pathlib import Path

from examples.common import print_header, print_kv, print_section
from stormlog import CPUMemoryTracker


def _load_batch() -> list[list[int]]:
    batch = [list(range(20_000)) for _ in range(8)]
    time.sleep(0.05)
    return batch


def _forward_pass(batch: list[list[int]]) -> list[int]:
    outputs = [sum(chunk) for chunk in batch]
    time.sleep(0.05)
    return outputs


def _backward_pass(outputs: list[int]) -> list[float]:
    gradients = [value / 1_000.0 for value in outputs]
    time.sleep(0.05)
    return gradients


def main() -> None:
    print_header("Stormlog - Phase Tracking Demo")
    print_section("Tracking")

    tracker = CPUMemoryTracker(sampling_interval=0.1, max_events=2_000)
    tracker.start_tracking()

    try:
        with tracker.phase("train", metadata={"epochs": 2}):
            for epoch in range(2):
                with tracker.phase("epoch", metadata={"epoch": epoch}):
                    with tracker.phase("load_batch"):
                        batch = _load_batch()
                    with tracker.phase("forward"):
                        outputs = _forward_pass(batch)
                    with tracker.phase("backward"):
                        _ = _backward_pass(outputs)
        time.sleep(0.2)
    finally:
        tracker.stop_tracking()

    output_dir = Path("artifacts/phase_tracking_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase_track.json"
    tracker.export_events(str(json_path), format="json")

    stats = tracker.get_statistics()
    print_section("Summary")
    print_kv("Total events", stats.get("total_events", 0))
    print_kv("Dropped events", stats.get("history_dropped_events", 0))
    print_kv("JSON export", json_path)
    print(
        "Next step: run `gpumemprof analyze artifacts/phase_tracking_demo/phase_track.json` "
        "to inspect phase-aware summaries."
    )


if __name__ == "__main__":
    main()
