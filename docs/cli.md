[← Back to main docs](index.md)

# Command Line Guide

Stormlog currently exposes three console scripts:

- `gpumemprof`
- `tfmemprof`
- `stormlog`

Use `gpumemprof` and `tfmemprof` for automation. Use `stormlog` when you want the Textual TUI.

If you want task-oriented operational recipes instead of option-by-option
guidance, use the [Production Cookbook](cookbook/index.md), especially
[Always-on Tracking](cookbook/always_on.md),
[Incident Playbooks](cookbook/incidents.md), and
[Distributed Diagnostics Recipes](cookbook/distributed.md).

## Verify the installed commands

```bash
gpumemprof --help
tfmemprof --help
```

If you are working from a repository checkout, `pip install -e .` also exposes
the source-only `examples/` package used in a few release-validation flows.

Install and launch the TUI with the current dependency set:

```bash
pip install "stormlog[tui,torch]"
stormlog
```

## `gpumemprof`

The current command groups are:

- `info`
- `monitor`
- `track`
- `analyze`
- `diagnose`

### Inspect environment

```bash
gpumemprof info
gpumemprof info --device 0 --detailed
```

`gpumemprof info` still reports the active PyTorch runtime first. When no
supported PyTorch GPU runtime is active, it now falls back to a best-effort host
GPU hardware probe so the command can show detected device names separately from
runtime availability. Supported PyTorch GPU runtimes remain NVIDIA CUDA, AMD
ROCm-backed PyTorch on Linux, and Apple MPS. In that unsupported-runtime mode,
`--device` is ignored because it only applies to an active PyTorch GPU runtime.

### Capture a bounded monitoring window

```bash
gpumemprof monitor --duration 30 --interval 0.5 --output monitor.csv --format csv
gpumemprof monitor --duration 30 --interval 0.5 --output monitor.json --format json
```

### Track events over time

```bash
gpumemprof track --duration 30 --interval 0.5 --output track.json --format json
gpumemprof track --warning-threshold 75 --critical-threshold 90 --output alerts.csv
gpumemprof track --job-id train-42 --rank 1 --local-rank 1 --world-size 8 --output rank1.json --format json
gpumemprof track --telemetry-sink-dir ./live_sink --telemetry-rollover-mb 32 --telemetry-retention-total-mb 256
```

Every `gpumemprof track` run now creates exactly one session identity. The
session begins after tracker startup succeeds and before the first record is
persisted, and it is marked `completed` only after clean shutdown finalization
finishes.

For long-running tracking sessions, Stormlog now degrades gracefully when a
collector becomes unhealthy:

- the tracked workload keeps running
- exported telemetry includes `collector_degraded` / `collector_recovered` events
- per-event metadata marks partial or unhealthy collector state
- retries use bounded exponential backoff instead of crashing the tracker loop

For always-on sessions, `gpumemprof track` can also stream append-only
telemetry into a sink directory during the run instead of waiting for shutdown.
The sink writes JSONL segments plus a manifest, rolls segments when they hit the
configured size limit, and prunes the oldest closed segments to stay within the
retention budget.

The sink manifest also keeps a session ledger, so multiple runs can safely
share the same sink directory without merging captures. If a previous run was
still `running` when the process died, the next startup recovers it as
`interrupted` and starts a fresh session for the new run.

Useful sink options:

- `--telemetry-sink-dir`
- `--telemetry-flush-seconds`
- `--telemetry-rollover-mb`
- `--telemetry-retention-files`
- `--telemetry-retention-total-mb`

The always-on qualification harness assumes the default sink settings:

- flush every `2.0s`
- roll segments at `64 MB`
- retain at most `8` files
- retain at most `512 MB` total

When you inspect `track` output after a long run, look for these diagnostics in
the JSON or CLI summary:

- `rollover_count`
- `pruned_segment_count`
- `pruned_bytes`
- `final_retained_files`
- `final_retained_bytes`
- `history_retained_*`
- `history_dropped_*`

Optional OOM flight-recorder support:

```bash
gpumemprof track \
  --oom-flight-recorder \
  --oom-dump-dir ./oom_dumps \
  --oom-max-dumps 10 \
  --oom-max-total-mb 1024 \
  --output track.json --format json
```

### Analyze saved telemetry

```bash
gpumemprof analyze track.json --format txt --output analysis.txt
gpumemprof analyze track.json --visualization --plot-dir plots
gpumemprof analyze ./live_sink
gpumemprof analyze ./live_sink --session-id 2b30f4a4-7d2d-48f7-a9f6-7d40c14eb95e
```

`gpumemprof analyze` uses a positional input file. It can now read a normal JSON
telemetry export, a sink JSONL segment, or a sink directory containing the
current and rolled append-only outputs. If you add `--visualization`, plots are
written to the directory passed via `--plot-dir` or to `plots/` by default.

When multiple sessions are present, `gpumemprof analyze` selects:

1. the newest `completed` session
2. otherwise the newest `interrupted` session
3. otherwise the newest `incomplete` session

Use `--session-id` to analyze a specific capture instead of the default one.

For always-on deployment posture and incident response checklists, continue with
[Always-on Tracking](cookbook/always_on.md) and
[Incident Playbooks](cookbook/incidents.md).

### Produce a diagnose bundle

```bash
gpumemprof diagnose --duration 5 --interval 0.5 --output ./diag_bundle
gpumemprof diagnose --duration 0 --output ./diag_bundle_quick
gpumemprof diagnose --native-history --duration 0 --output ./diag_bundle_native
```

Use `--duration 0` when you want a fast artifact bundle without a new tracking window.

For task-oriented recipes that combine `track`, `analyze`, `diagnose`, and the
TUI into one workflow, continue with
[Incident Playbooks](cookbook/incidents.md) and
[Distributed Diagnostics Recipes](cookbook/distributed.md).

Each standalone diagnose bundle also owns its own session id. The bundle
manifest records whether the run finished `completed` or was left
`incomplete`, and synthesized timeline telemetry inherits that same session id
when reloaded later.

`--native-history` is a CUDA-only debug mode. It records allocator history for
the current `gpumemprof diagnose` process, then writes native snapshot artifacts
such as `cuda_allocator_snapshot.pickle`,
`cuda_allocator_state_history.html`,
`cuda_allocator_state_history_annotated.html`, and tensor-attribution JSON
alongside the normal diagnose bundle files. The annotated HTML is the
Stormlog-native view that exposes the timeline trace, segment explorer, and
active-memory table in one file. For a maintained workflow example of that
artifact, continue with [PyTorch Production Recipes](cookbook/pytorch.md). On
MPS, ROCm, or CPU-only runtimes, the command fails explicitly instead of
pretending support.

## `tfmemprof`

The current command groups are:

- `info`
- `monitor`
- `track`
- `analyze`
- `diagnose`

### Inspect environment

```bash
tfmemprof info
```

### Monitor TensorFlow memory usage

```bash
tfmemprof monitor --interval 0.5 --duration 30 --output tf_monitor.json
tfmemprof monitor --interval 0.5 --duration 30 --threshold 4096 --device /GPU:0 --output tf_monitor_threshold.json
```

For CPU-only TensorFlow or when the GPU backend is unavailable, use
`--device /CPU:0`:

```bash
tfmemprof monitor --interval 0.5 --duration 30 --device /CPU:0 --output tf_monitor.json
tfmemprof track --interval 0.5 --threshold 4096 --device /CPU:0 --output tf_track.json
```

### Track TensorFlow memory usage

```bash
tfmemprof track --interval 0.5 --threshold 4096 --output tf_track.json
tfmemprof track --interval 0.5 --threshold 4096 --job-id train-42 --rank 3 --local-rank 1 --world-size 8 --output tf_rank3.json
tfmemprof track --interval 0.5 --threshold 4096 --output tf_track.json --telemetry-sink-dir ./tf_live_sink
```

`tfmemprof track` follows the same degraded-mode rules as the PyTorch tracker:
collector failures pause new sample emission, status events remain visible in the
artifact stream, and normal sampling resumes automatically after recovery.
The same append-only sink options are available when you need bounded,
interrupt-tolerant TensorFlow telemetry during a long-running session.

TensorFlow tracking also keeps only a bounded recent history in memory. The
current CLI output and JSON exports surface the retained vs dropped sample,
event, and alert counts so long-running jobs can distinguish expected eviction
from a silent memory-growth regression.

The same session rules apply to TensorFlow tracking:

- one session id per `tfmemprof track` run
- sink recovery marks old running sessions as `interrupted`
- loaders and diagnostics separate same-host runs by `session_id`, not by job or rank alone

### Analyze TensorFlow results

```bash
tfmemprof analyze --input tf_monitor.json --detect-leaks --optimize
tfmemprof analyze --input tf_monitor.json --detect-leaks --optimize --visualize --report tf_report.txt
```

Unlike `gpumemprof analyze`, the TensorFlow analyzer uses `--input`.

### Produce a diagnose bundle

```bash
tfmemprof diagnose --duration 5 --interval 0.5 --output ./tf_diag
tfmemprof diagnose --duration 0 --output ./tf_diag_quick
```

## TUI launch

```bash
pip install "stormlog[tui,torch]"
stormlog
```

Inside the TUI, the `CLI & Actions` tab exposes quick actions for:

- `gpumemprof info`
- `gpumemprof monitor`
- `tfmemprof monitor`
- `gpumemprof diagnose`
- sample workloads
- OOM scenario runner
- capability matrix smoke run

## Release-validation shortcuts

**Source checkout only.** These commands require the `examples/` package from a
repository clone:

```bash
python -m examples.cli.quickstart
python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
```

**Pip users** should use this CLI-only sequence instead:

```bash
gpumemprof info
gpumemprof track --duration 2 --interval 0.5 --output track.json --format json
gpumemprof analyze track.json --format txt --output analysis.txt
gpumemprof diagnose --duration 0 --output ./diag

tfmemprof info
tfmemprof diagnose --duration 0 --output ./tf_diag
```

## Choosing the right command

### Use `monitor` when

- you want a bounded sample window
- you only need a simple CSV or JSON output

### Use `track` when

- you want event streams and alert thresholds
- you want later exports or distributed identity fields
- you want a reconstructable session that owns its sink, diagnose, or OOM artifacts

### Use `analyze` when

- you already have saved telemetry
- you want a report or plot output

### Use `diagnose` when

- you need a portable artifact bundle to archive or share
- you plan to inspect the output later in the TUI diagnostics flow

---

[← Back to main docs](index.md)
