[← Back to Production Cookbook](index.md)

# Incident Playbooks

Use these checklists when you already have a failure mode and need the shortest
path to the next useful artifact or decision.

Validated against the current `release/dev` branch on April 7, 2026.

Audience: incident responders, ML engineers.
Difficulty: intermediate.

## Prerequisites

- install the package first with [Installation](../installation.md)
- use `pip install "stormlog[torch]"` for `gpumemprof` paths
- use `pip install "stormlog[tf]"` for `tfmemprof` paths
- use `pip install "stormlog[tui,torch]"` if you want to pivot into the TUI diagnostics workflow
- use [Command Line Guide](../cli.md) and [Troubleshooting Guide](../troubleshooting.md) for command or environment reference
- enough disk space for new diagnose or telemetry artifacts
- the relevant CLI entrypoint available in the current environment
- a clear incident label before you start: OOM, hidden-memory gap, degraded collector, or retention budget

Success signal:

- the incident ends with a saved artifact plus a concrete follow-on recipe

## OOM incident

### Use this when

- the process already failed with an out-of-memory error
- you need a portable artifact bundle, not just console logs

### Immediate actions

```bash
gpumemprof diagnose --duration 0 --output ./diag_bundle
```

```bash
gpumemprof diagnose --native-history --duration 0 --output ./diag_bundle_native
```

### Next steps

1. Keep the standard diagnose bundle even if you also collect native CUDA history.
2. Check whether the bundle includes the owning `session_id` and a clean or incomplete session status.
3. If the workload is PyTorch CUDA, inspect the native-history artifacts before changing allocator settings.
4. If the issue is recurring, enable the OOM flight-recorder path from the
   [PyTorch Production Recipes](pytorch.md).

## Hidden-memory-gap incident

### Use this when

- allocated memory does not explain the observed peak
- the report includes `gap_analysis`
- the multi-rank case suggests communication-driven spikes

### Immediate actions

```bash
gpumemprof analyze ./track.json --format txt --output ./analysis.txt
```

```bash
tfmemprof analyze --input ./tf_monitor.json --detect-leaks --optimize --report ./tf_report.txt
```

### Next steps

1. Start with `gap_analysis` before assuming a leak.
2. If `collective_attribution` is populated in a multi-rank run, treat communication or synchronization as a first-class cause.
3. Load the artifacts into the TUI Diagnostics tab if you need rank filtering and timeline focus.
4. If the workload is distributed, move to the
   [Distributed Diagnostics Recipes](distributed.md).

## Degraded collector incident

### Use this when

- `collector_failure_event_count` is non-zero
- collector health is not `healthy`
- status events show `collector_degraded` or `collector_recovered`

### Immediate actions

```bash
gpumemprof track \
  --duration 30 \
  --interval 0.5 \
  --output ./track.json \
  --format json
```

```bash
tfmemprof track --interval 0.5 --threshold 4096 --device /CPU:0 --output ./tf_track.json
```

### Next steps

1. Confirm that the run stayed alive and emitted status events instead of synthetic zero-valued samples.
2. Treat any non-zero collector failure count as actionable even if the final session completed cleanly.
3. If the collector only fails in long runs, move to the
   [Always-on Tracking](always_on.md) page and qualify the deployment under sink retention.
4. If the failure is reproducible in a short window, archive the bounded artifact and compare it against a clean run.

## Retention-budget incident

### Use this when

- retained files or bytes exceed expectations
- rollover or pruning looks abnormal
- long runs create more artifact churn than the deployment budget allows

### Immediate actions

```bash
python -m examples.cli.benchmark_harness \
  --profile pr \
  --mode soak \
  --output artifacts/benchmarks/latest_v0.4.json
```

This command is source-checkout only. If you installed from PyPI, use a bounded
artifact path first:

```bash
gpumemprof track --duration 30 --interval 0.5 --output ./track.json --format json
```

### Next steps

1. Inspect `rollover_count`, `pruned_segment_count`, `pruned_bytes`, `final_retained_files`, and `final_retained_bytes` together.
2. Tighten retention and rollover settings before reducing sampling fidelity.
3. If the issue needs a repeatable gate, move to
   [CI and Release Qualification](ci_release.md).

## Troubleshooting

### Symptom: the incident does not match one checklist cleanly

Likely cause: the first artifact is too small or the failure mode is mixed.
Fix: capture a bounded `track` artifact first, then reclassify based on the saved data.
Verify: the next decision is driven by saved evidence rather than a log fragment.

---

[← Back to Production Cookbook](index.md)
