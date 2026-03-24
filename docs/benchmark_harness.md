[← Back to main docs](index.md)

# Benchmark Harness (v0.3)

> **Source checkout only.** `python -m examples.cli.benchmark_harness` requires
> the repository `examples/` package and `docs/benchmarks/`. It is not shipped
> in the PyPI package.

This benchmark harness measures CPU tracker overhead and artifact growth. In v0.3
it supports two gate modes:

- `budget`: compare current metrics to absolute max thresholds
- `regression`: compare current metrics to a checked-in baseline plus allowed deltas

## Run the Harness

```bash
python -m examples.cli.benchmark_harness \
  --iterations 200 \
  --output artifacts/benchmarks/latest.json
```

## Enforce Regression Gate

```bash
python -m examples.cli.benchmark_harness \
  --check \
  --gate-mode regression \
  --iterations 5000 \
  --baseline docs/benchmarks/v0.3_baseline.json \
  --tolerances docs/benchmarks/v0.3_tolerances.json \
  --output artifacts/benchmarks/latest.json
```

Use this mode for the v0.3 CI memory regression gate. The regression check uses
`--iterations 5000` so the baseline signal is less noisy than the lighter v0.2
budget check.

## Enforce Budgets

```bash
python -m examples.cli.benchmark_harness \
  --check \
  --gate-mode budget \
  --iterations 200 \
  --budgets docs/benchmarks/v0.2_budgets.json \
  --output artifacts/benchmarks/latest.json
```

Use `--check` in budget mode when you want the older absolute-threshold policy.

## What It Measures

- `runtime_overhead_pct`: wall-time overhead of a tracked run vs an unprofiled run.
- `cpu_overhead_pct`: CPU-time overhead of a tracked run vs an unprofiled run.
- `sampling_impact_pct`: extra wall-time cost of default sampling vs lower-frequency sampling.
- `artifact_growth_bytes`: additional artifact size from the tracked run vs the unprofiled run.

The current implementation uses `CPUMemoryTracker` and a deterministic CPU workload in `examples/cli/benchmark_harness.py`. Treat it as a budget harness for tracking overhead, not as a full-framework performance benchmark.

## Output Format

The JSON report includes:

- `gate_mode`: which policy was evaluated (`budget` or `regression`).
- `config`: benchmark configuration and paths.
- `scenarios`: per-scenario runtime, CPU time, event count, and artifact size.
- `metrics`: computed deltas used for gating.
- `budget_checks`: per-metric value/max/passed results in budget mode.
- `baseline`: checked-in baseline config and metrics in regression mode.
- `tolerances`: allowed positive deltas in regression mode.
- `regression_checks`: per-metric current/baseline/delta/allowed/passed results.
- `passed`: overall gate status.

## Baseline And Tolerance Files

The v0.3 regression gate reads:

- `docs/benchmarks/v0.3_baseline.json`
- `docs/benchmarks/v0.3_tolerances.json`

Update these files in versioned commits when the accepted performance envelope
changes. Keep baseline updates explicit: run the harness with the same config as
the CI job, inspect the new metrics, then commit the baseline or tolerance
change separately from unrelated code.

The older absolute thresholds still live in:

`docs/benchmarks/v0.2_budgets.json`
