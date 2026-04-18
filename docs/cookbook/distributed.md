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

## Recipe: validate a reference `torchrun` DDP run on Jarvis

Use this when you want one real multi-GPU training run based on the official
PyTorch DDP tutorial pattern, not a manually stitched set of rank-local
captures.

This workflow was validated against:

- PyTorch DDP tutorial: `https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html`
- PyTorch tutorial-series example repo:
  `https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series`
- local adaptation: `examples.scenarios.torchrun_ddp_reference`

Validated environment:

- Jarvis container instance
- `2xL4`
- PyTorch template
- single node with `torchrun`

Assume the source checkout is already present on the instance at
`/home/gpu-memory-profiler`.

```bash
jl create \
  --gpu L4 \
  --num-gpus 2 \
  --template pytorch \
  --region IN2 \
  --storage 80 \
  --name stormlog-ddp-reference \
  --yes \
  --json
```

Prepare the project environment on the instance:

```bash
cd /home/gpu-memory-profiler
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Run the reference training job with `torchrun`:

```bash
cd /home/gpu-memory-profiler
. .venv/bin/activate
PYTHONPATH=/home/gpu-memory-profiler \
OMP_NUM_THREADS=1 \
python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  -m examples.scenarios.torchrun_ddp_reference \
  --epochs 2 \
  --batch-size 128 \
  --dataset-size 4096 \
  --interval 0.1 \
  --job-id jarvis-torchrun-reference \
  --output-dir artifacts/jarvis_torchrun_reference \
  2>&1 | tee artifacts/jarvis_torchrun_reference/run.log
```

Expected console output shape:

- one loss line per rank per epoch
- rank-local and global loss values
- a final `Reference summary saved to .../ddp_reference_summary.json`

Expected artifacts:

- `artifacts/jarvis_torchrun_reference/ddp_reference_summary.json`
- `artifacts/jarvis_torchrun_reference/rank0/telemetry_sink/`
- `artifacts/jarvis_torchrun_reference/rank1/telemetry_sink/`
- `artifacts/jarvis_torchrun_reference/reference_checkpoint.pt`
- `artifacts/jarvis_torchrun_reference/run.log`

The validated reference run produced:

- `world_size = 2`
- `rank_summaries = 2`
- `12` sample events per rank
- `133` `phase_enter` and `133` `phase_exit` events per rank

Download the artifact root locally, then load both rank sinks together in the
TUI Diagnostics tab:

```bash
stormlog
```

Then:

1. Open `Diagnostics`.
2. Enter the two sink paths as a comma-separated list.
3. Click `Load Artifacts`.
4. Leave session selection on `auto`.
5. Confirm `present_ranks` shows `0,1`.

For this reference run, Diagnostics selected a merged synthetic session and
reported:

- `present_ranks = [0, 1]`
- `expected_ranks = [0, 1]`
- `missing_ranks = []`
- one diagnostics row per rank

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
