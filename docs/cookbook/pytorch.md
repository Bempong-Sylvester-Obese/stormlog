[← Back to Production Cookbook](index.md)

# PyTorch Production Recipes

Use these recipes when the workload is owned by PyTorch and you need a fast path
from a memory incident to a saved artifact and an actionable next step.

Validated against the current `release/dev` branch on April 7, 2026.

Audience: ML engineers, incident responders.
Difficulty: intermediate.

## Prerequisites

- install the package first with [Installation](../installation.md)
- use `pip install "stormlog[torch]"` for the PyTorch CLI and tracker paths
- use `pip install "stormlog[tui,torch]"` if you want the TUI diagnostics flow
- use a PyTorch runtime when you need `GPUMemoryProfiler` or CUDA-specific OOM evidence
- use the [Usage Guide](../usage.md) if you need API-level reference instead of CLI-first triage

Success signal:

- you leave the incident with at least one saved artifact or report
- the chosen workflow ends in a concrete next step, not just console output

## Choose the first PyTorch recipe

| If the main goal is... | Start with... |
| --- | --- |
| capture a short, shareable timeline | bounded `track` |
| save a portable diagnostic bundle fast | `diagnose --duration 0` |
| capture CUDA allocator-history evidence | `diagnose --native-history` |
| rehearse the OOM flow safely in a checkout | `examples.scenarios.oom_flight_recorder_scenario` |

## Recipe: capture a bounded telemetry window

```bash
gpumemprof track \
  --duration 30 \
  --interval 0.5 \
  --output ./track.json \
  --format json
```

Use this when you want a short capture that you can archive, share, or analyze
without keeping a sink directory around.

## Recipe: export a diagnose bundle

```bash
gpumemprof diagnose --duration 0 --output ./diag_bundle
```

Use `--duration 0` when you want the bundle structure and current runtime
context quickly, not a new long sampling window.

## Recipe: capture CUDA-native OOM evidence

```bash
gpumemprof diagnose --native-history --duration 0 --output ./diag_bundle_native
```

This is the CUDA-only path for allocator-history artifacts such as native
snapshots and state-history files.

Do not use this path on CPU-only or non-CUDA hosts. Fall back to the standard
diagnose bundle there.

## Recipe: turn saved telemetry into a report

```bash
gpumemprof analyze ./track.json --format txt --output ./analysis.txt
```

Use `--visualization --plot-dir ./plots` when you also want saved plots.

## Recipe: rehearse the OOM workflow safely from a source checkout

```bash
python -m examples.scenarios.oom_flight_recorder_scenario --mode simulated
```

This is source-checkout only. Pip installs do not include `examples/`.

## What to look for in the report

- `critical_issues`
- `high_impact_insights`
- `recommendations`
- `optimization_score`
- `gap_analysis` when telemetry events are available
- `collective_attribution` when hidden-memory spikes align with communication signals

## What to do next

- If `critical_issues` or high-priority recommendations point to growth over
  time, move to the [Incident Playbooks](incidents.md) leak and growth path.
- If `gap_analysis` is populated, move to the hidden-memory-gap checklist in
  [Incident Playbooks](incidents.md).
- If `collective_attribution` is populated and the workload is multi-rank, move
  to the [Distributed Diagnostics Recipes](distributed.md).
- If the main need is a long-running operational deployment, move to
  [Always-on Tracking](always_on.md).

## Troubleshooting

### Symptom: `diagnose --native-history` fails immediately

Likely cause: the runtime is not CUDA-backed.
Fix: use the standard diagnose bundle instead.
Verify: the regular diagnose command completes and writes a manifest.

### Symptom: the report suggests hidden memory but not a leak

Likely cause: the peak is not fully explained by allocated-memory telemetry.
Fix: inspect `gap_analysis` and `collective_attribution` before changing allocator settings.
Verify: the next step comes from the hidden-memory-gap path, not generic leak tuning.

---

[← Back to Production Cookbook](index.md)
