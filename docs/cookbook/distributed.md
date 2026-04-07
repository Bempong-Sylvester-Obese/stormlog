[← Back to Production Cookbook](index.md)

# Distributed Diagnostics Recipes

Use this page when you need to preserve per-rank identity during capture and
rebuild a rank-aware timeline later.

Audience: distributed-training owners, incident responders.
Difficulty: advanced.

## Prerequisites

- install the package first with [Installation](../installation.md)
- use `pip install "stormlog[torch]"` for PyTorch rank capture
- use `pip install "stormlog[tf]"` for TensorFlow rank capture
- use `pip install "stormlog[tui,torch]"` if you want the TUI Diagnostics workflow from a pip install
- use [TUI Guide](../tui.md) and [Command Line Guide](../cli.md) if you need UI or CLI reference details
- the artifact paths for each rank are writable and distinct
- rank metadata is either inferred from the environment or passed explicitly
- the TUI extra is installed if you want the interactive Diagnostics workflow

Success signal:

- each rank produces its own artifact with rank identity intact
- the TUI Diagnostics tab can load multiple rank files without flattening them

## Choose the distributed path

| If the job is... | Start with... |
| --- | --- |
| PyTorch rank capture | rank-aware `gpumemprof track` |
| TensorFlow rank capture | rank-aware `tfmemprof track` |
| artifact triage after capture | `stormlog` Diagnostics tab |

## When this is the right recipe

- you need one artifact per rank
- you want `job_id`, `rank`, `local_rank`, and `world_size` recorded explicitly
- you need TUI diagnostics to keep ranks separate instead of flattening them
- you want hidden-memory-gap or collective-attribution analysis with more than one rank

## Recipe: capture rank-aware PyTorch artifacts

```bash
gpumemprof track \
  --duration 30 \
  --interval 0.5 \
  --job-id train-42 \
  --rank 0 \
  --local-rank 0 \
  --world-size 2 \
  --output ./rank0.json \
  --format json
```

```bash
gpumemprof track \
  --duration 30 \
  --interval 0.5 \
  --job-id train-42 \
  --rank 1 \
  --local-rank 1 \
  --world-size 2 \
  --output ./rank1.json \
  --format json
```

## Recipe: capture rank-aware TensorFlow artifacts

```bash
tfmemprof track \
  --interval 0.5 \
  --threshold 4096 \
  --device /CPU:0 \
  --job-id train-42 \
  --rank 0 \
  --local-rank 0 \
  --world-size 2 \
  --output ./tf_rank0.json
```

```bash
tfmemprof track \
  --interval 0.5 \
  --threshold 4096 \
  --device /CPU:0 \
  --job-id train-42 \
  --rank 1 \
  --local-rank 1 \
  --world-size 2 \
  --output ./tf_rank1.json
```

Stop each TensorFlow rank cleanly with `Ctrl+C` after tracking has started so
the per-rank output file is flushed before exit.

Keep the same `job_id` across every rank-local capture from one distributed
run. Diagnostics uses that shared job identity to auto-select a merged
cross-rank session when you load the artifacts together.

## Recipe: load multiple rank artifacts in the TUI

```bash
stormlog
```

Then:

1. Open `Diagnostics`.
2. Enter the artifact paths as a comma-separated list.
3. Click `Load Artifacts`.
4. Leave session selection on `auto` or `default` first. With a shared `job_id`,
   Diagnostics selects the merged cross-rank session automatically.
5. Choose an individual `session_id` only when you want to isolate one raw
   rank-local artifact.
6. Apply a rank filter such as `all` or `0,1`.

## What to look for

- `cross_rank_analysis` in PyTorch optimization reports when more than one rank is present
- rank-aware timeline differences in the TUI diagnostics pane
- `collective_attribution` when communication phases align with hidden-memory spikes
- session separation by `session_id` when the same sink directory or host is reused

## What to do next

- If one rank is the first cause, isolate that rank's artifact and analyze it independently.
- If all ranks spike together and `collective_attribution` is populated, treat the issue as a communication or synchronization candidate before changing model code.
- If the problem is operational rather than rank-local, move to
  [Always-on Tracking](always_on.md).

## Troubleshooting

### Symptom: only one rank appears in diagnostics

Likely cause: the wrong artifact set or session was loaded.
Fix: load every rank artifact together, then target the intended session before refreshing.
Verify: `present_ranks` matches the expected rank set.

### Symptom: ranks are present but the first cause is unclear

Likely cause: the issue is synchronized across ranks.
Fix: inspect `cross_rank_analysis` and `collective_attribution` before isolating one rank.
Verify: the next action comes from a rank-local or communication-attributed explanation, not guesswork.

---

[← Back to Production Cookbook](index.md)
