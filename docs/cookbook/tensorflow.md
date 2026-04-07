[← Back to Production Cookbook](index.md)

# TensorFlow Production Recipes

Use these recipes when the runtime is TensorFlow and you need production-safe
capture, analysis, and diagnosis flows that match the current `tfmemprof`
behavior.

Validated against the current `release/dev` branch on April 7, 2026.

Audience: ML engineers, release owners.
Difficulty: intermediate.

## Prerequisites

- install the package first with [Installation](../installation.md)
- use `pip install "stormlog[tf]"` for the TensorFlow CLI paths
- use [Command Line Guide](../cli.md) if you need per-flag reference
- pick `/CPU:0` or `/GPU:0` explicitly for the current runtime

Success signal:

- a monitor, track, or diagnose artifact is written successfully
- the analyzer returns a report with a clear next action

## Choose the first TensorFlow recipe

| If the main goal is... | Start with... |
| --- | --- |
| capture a bounded sample window | `monitor` |
| keep an event stream and session status | `track` |
| get a report with leak and optimization signals | `analyze` |
| save a portable bundle fast | `diagnose --duration 0` |

## Recipe: capture a bounded telemetry window

```bash
tfmemprof monitor --interval 0.5 --duration 30 --device /CPU:0 --output ./tf_monitor.json
```

Switch to `/GPU:0` when the TensorFlow runtime exposes a GPU device.

## Recipe: track TensorFlow memory over time

```bash
tfmemprof track \
  --interval 0.5 \
  --threshold 4096 \
  --device /CPU:0 \
  --output ./tf_track.json
```

Use `track` when you need retained vs dropped history counters, session status,
and an event stream you can reload later.

Stop the command cleanly with `Ctrl+C` so the output file is flushed before the
process exits.

## Recipe: run TensorFlow analysis

```bash
tfmemprof analyze --input ./tf_monitor.json --detect-leaks --optimize --report ./tf_report.txt
```

The current TensorFlow analyzer uses `--input`, not the positional-input style
from `gpumemprof analyze`.

## Recipe: produce a diagnose bundle

```bash
tfmemprof diagnose --duration 0 --output ./tf_diag_bundle
```

## Recipe: validate the end-to-end TensorFlow flow from a source checkout

```bash
python -m examples.scenarios.tf_end_to_end_scenario
```

This is source-checkout only. Pip installs do not include `examples/`.

## What to look for in the results

- leak findings from `--detect-leaks`
- optimization recommendations from `--optimize`
- `collector_failure_event_count`
- `session_status`
- `gap_analysis` when telemetry events are available
- `collective_attribution` when cross-rank communication likely explains hidden-memory spikes

## What to do next

- If leak detection reports high-severity issues, move to the memory-growth path
  in [Incident Playbooks](incidents.md).
- If `collector_failure_event_count` is non-zero, use the degraded-collector
  checklist in [Incident Playbooks](incidents.md).
- If the run is part of a larger distributed job, move to the
  [Distributed Diagnostics Recipes](distributed.md).
- If the deployment is long-running, move to [Always-on Tracking](always_on.md).

## Troubleshooting

### Symptom: `track` stops without writing the output file

Likely cause: the process was interrupted before the tracker reached its normal shutdown path.
Fix: wait until tracking has started, then stop it cleanly with `Ctrl+C`.
Verify: the output file is written and `session_status` is present.

### Symptom: `analyze` reports no GPU

Likely cause: the TensorFlow runtime is CPU-only or the selected device is unavailable.
Fix: rerun with `--device /CPU:0` or fix the runtime environment first.
Verify: `tfmemprof info` and the chosen capture command agree on the active device.

---

[← Back to Production Cookbook](index.md)
