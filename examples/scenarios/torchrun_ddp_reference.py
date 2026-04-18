"""Reference DDP training example derived from the official PyTorch tutorials.

This module keeps the single-node `torchrun` launch pattern from the official
PyTorch DDP material and layers Stormlog telemetry on top of it.

Primary upstream references:
- https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
- https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data.distributed import DistributedSampler
except ImportError as exc:  # pragma: no cover - exercised in runtime envs
    raise RuntimeError(
        "PyTorch is required for this example. Install with `pip install 'stormlog[torch]'`."
    ) from exc

from stormlog.telemetry_sink import TelemetrySinkConfig
from stormlog.tracker import MemoryTracker

_INPUT_DIM = 1024
_HIDDEN_DIM = 2048
_NUM_CLASSES = 10


@dataclass(frozen=True)
class RankContext:
    rank: int
    local_rank: int
    world_size: int


@dataclass(frozen=True)
class ArtifactPaths:
    root_dir: Path
    rank_dir: Path
    telemetry_sink_dir: Path
    rank_summary_path: Path
    root_summary_path: Path
    checkpoint_path: Path


class SyntheticClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        *,
        size: int,
        input_dim: int,
        num_classes: int,
        seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self._features = torch.randn(size, input_dim, generator=generator)
        self._targets = torch.randint(
            low=0,
            high=num_classes,
            size=(size,),
            generator=generator,
        )

    def __len__(self) -> int:
        return int(self._targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._features[index], self._targets[index]


class TutorialNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference single-node DDP training with torchrun and Stormlog.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Per-rank batch size.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=4096,
        help="Number of synthetic samples each rank sees before sharding.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Stormlog sampling interval in seconds.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default="pytorch-torchrun-reference",
        help="Shared job identifier recorded in telemetry.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/pytorch_torchrun_reference",
        help="Directory for rank-local telemetry and summary files.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Checkpoint cadence in epochs for rank 0.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Global random seed.")
    return parser.parse_args()


def _rank_context_from_env(env: dict[str, str]) -> RankContext:
    missing = [
        key for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if key not in env
    ]
    if missing:
        raise RuntimeError(
            "This example must be launched with torchrun. Missing env vars: "
            + ", ".join(sorted(missing))
        )
    return RankContext(
        rank=int(env["RANK"]),
        local_rank=int(env["LOCAL_RANK"]),
        world_size=int(env["WORLD_SIZE"]),
    )


def _artifact_paths(output_dir: Path, rank_context: RankContext) -> ArtifactPaths:
    rank_dir = output_dir / f"rank{rank_context.rank}"
    telemetry_sink_dir = rank_dir / "telemetry_sink"
    return ArtifactPaths(
        root_dir=output_dir,
        rank_dir=rank_dir,
        telemetry_sink_dir=telemetry_sink_dir,
        rank_summary_path=rank_dir / "rank_summary.json",
        root_summary_path=output_dir / "ddp_reference_summary.json",
        checkpoint_path=output_dir / "reference_checkpoint.pt",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _setup_ddp(rank_context: RankContext) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this reference DDP example.")
    torch.cuda.set_device(rank_context.local_rank)
    dist.init_process_group(backend="nccl")
    return torch.device(f"cuda:{rank_context.local_rank}")


def _build_dataloader(args: argparse.Namespace) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = SyntheticClassificationDataset(
        size=args.dataset_size,
        input_dim=_INPUT_DIM,
        num_classes=_NUM_CLASSES,
        seed=args.seed,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(dataset, shuffle=True),
        pin_memory=True,
        shuffle=False,
    )


def _build_tracker(
    args: argparse.Namespace,
    *,
    rank_context: RankContext,
    telemetry_sink_dir: Path,
) -> MemoryTracker:
    telemetry_sink_dir.mkdir(parents=True, exist_ok=True)
    telemetry_sink_config = TelemetrySinkConfig(
        root_dir=telemetry_sink_dir,
        flush_every_seconds=1.0,
        rollover_max_bytes=32 * 1024 * 1024,
        retention_max_files=4,
        retention_max_total_bytes=128 * 1024 * 1024,
    )
    return MemoryTracker(
        device=f"cuda:{rank_context.local_rank}",
        sampling_interval=args.interval,
        enable_alerts=True,
        job_id=args.job_id,
        telemetry_sink_config=telemetry_sink_config,
    )


def _load_training_objects() -> tuple[nn.Module, torch.optim.Optimizer]:
    model = TutorialNet(
        input_dim=_INPUT_DIM,
        hidden_dim=_HIDDEN_DIM,
        num_classes=_NUM_CLASSES,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def _save_checkpoint(checkpoint_path: Path, epoch_index: int, model: DDP) -> None:
    torch.save(
        {
            "epoch": epoch_index,
            "model_state": model.module.state_dict(),
        },
        checkpoint_path,
    )


def _train(
    *,
    args: argparse.Namespace,
    rank_context: RankContext,
    device: torch.device,
    tracker: MemoryTracker,
    artifact_paths: ArtifactPaths,
) -> dict[str, Any]:
    dataloader = _build_dataloader(args)
    model, optimizer = _load_training_objects()
    ddp_model = DDP(model.to(device), device_ids=[rank_context.local_rank])
    epoch_summaries: list[dict[str, Any]] = []

    with tracker.phase(
        "reference_run",
        metadata={
            "tutorial_source": "pytorch/examples distributed/ddp-tutorial-series/multigpu_torchrun.py",
            "rank": rank_context.rank,
            "world_size": rank_context.world_size,
            "epochs": args.epochs,
        },
    ):
        for epoch_index in range(args.epochs):
            dataloader.sampler.set_epoch(epoch_index)
            epoch_loss_total = 0.0
            step_count = 0

            with tracker.phase(
                "epoch",
                metadata={"epoch": epoch_index + 1, "steps": len(dataloader)},
            ):
                for step_index, (features, targets) in enumerate(dataloader, start=1):
                    with tracker.phase(
                        "batch",
                        metadata={"epoch": epoch_index + 1, "step": step_index},
                    ):
                        features = features.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)
                        with tracker.phase(
                            "forward",
                            metadata={"epoch": epoch_index + 1, "step": step_index},
                        ):
                            logits = ddp_model(features)
                            loss = F.cross_entropy(logits, targets)
                        with tracker.phase(
                            "backward",
                            metadata={"epoch": epoch_index + 1, "step": step_index},
                        ):
                            loss.backward()
                        with tracker.phase(
                            "optimizer_step",
                            metadata={"epoch": epoch_index + 1, "step": step_index},
                        ):
                            optimizer.step()

                        epoch_loss_total += float(loss.detach().cpu().item())
                        step_count += 1

            local_average_loss = epoch_loss_total / max(step_count, 1)
            reduced_loss = torch.tensor(local_average_loss, device=device)
            with tracker.phase(
                "epoch_loss_all_reduce",
                metadata={"epoch": epoch_index + 1},
            ):
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            global_average_loss = float(reduced_loss.item() / rank_context.world_size)

            epoch_summary = {
                "epoch": epoch_index + 1,
                "local_average_loss": local_average_loss,
                "global_average_loss": global_average_loss,
                "rank": rank_context.rank,
            }
            epoch_summaries.append(epoch_summary)
            print(
                f"[rank {rank_context.rank}] epoch {epoch_index + 1} "
                f"local_loss={local_average_loss:.6f} global_loss={global_average_loss:.6f}"
            )

            if rank_context.rank == 0 and (epoch_index + 1) % args.save_every == 0:
                _save_checkpoint(artifact_paths.checkpoint_path, epoch_index + 1, ddp_model)

    return {
        "rank": rank_context.rank,
        "local_rank": rank_context.local_rank,
        "world_size": rank_context.world_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dataset_size": args.dataset_size,
        "epoch_summaries": epoch_summaries,
        "telemetry_sink_dir": str(artifact_paths.telemetry_sink_dir),
        "checkpoint_path": (
            str(artifact_paths.checkpoint_path)
            if artifact_paths.checkpoint_path.exists()
            else None
        ),
    }


def main() -> None:
    args = _parse_args()
    rank_context = _rank_context_from_env(dict(os.environ))
    artifact_paths = _artifact_paths(Path(args.output_dir).resolve(), rank_context)
    artifact_paths.rank_dir.mkdir(parents=True, exist_ok=True)
    device = _setup_ddp(rank_context)
    tracker = _build_tracker(
        args,
        rank_context=rank_context,
        telemetry_sink_dir=artifact_paths.telemetry_sink_dir,
    )
    tracker.start_tracking()
    rank_summary: dict[str, Any] | None = None

    try:
        rank_summary = _train(
            args=args,
            rank_context=rank_context,
            device=device,
            tracker=tracker,
            artifact_paths=artifact_paths,
        )
    finally:
        tracker.stop_tracking()

    try:
        if rank_summary is not None:
            rank_summary["tracker_statistics"] = tracker.get_statistics()
            _write_json(artifact_paths.rank_summary_path, rank_summary)

            dist.barrier(device_ids=[rank_context.local_rank])
            if rank_context.rank == 0:
                rank_summaries = []
                for rank in range(rank_context.world_size):
                    summary_path = (
                        artifact_paths.root_dir / f"rank{rank}" / "rank_summary.json"
                    )
                    if summary_path.exists():
                        rank_summaries.append(
                            json.loads(summary_path.read_text(encoding="utf-8"))
                        )
                _write_json(
                    artifact_paths.root_summary_path,
                    {
                        "job_id": args.job_id,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "dataset_size": args.dataset_size,
                        "world_size": rank_context.world_size,
                        "artifact_root": str(artifact_paths.root_dir),
                        "rank_summaries": rank_summaries,
                        "tutorial_sources": [
                            "https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html",
                            "https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series",
                        ],
                    },
                )
                print(f"Reference summary saved to {artifact_paths.root_summary_path}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
