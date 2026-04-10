"""CNN training scenario on handwritten digits with Stormlog + W&B export."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
except ImportError as exc:  # pragma: no cover - exercised in runtime envs
    raise RuntimeError(
        "PyTorch is required for this example. Install with `pip install 'stormlog[torch]'`."
    ) from exc

try:
    import wandb
except ImportError as exc:  # pragma: no cover - exercised in runtime envs
    raise RuntimeError(
        "Weights & Biases is required for this example. "
        "Install with `pip install 'stormlog[wandb]'`."
    ) from exc

from stormlog.cpu_profiler import CPUMemoryTracker
from stormlog.telemetry_sink import TelemetrySinkConfig
from stormlog.tracker import MemoryTracker
from stormlog.wandb_integration import WandbExportConfig, export_tracking_run_to_wandb

KMNIST_CLASS_NAMES = [
    "o",
    "ki",
    "su",
    "tsu",
    "na",
    "ha",
    "ma",
    "ya",
    "re",
    "wo",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small CNN on handwritten characters with Stormlog + W&B",
    )
    parser.add_argument(
        "--dataset",
        choices=["kmnist", "mnist"],
        default="mnist",
        help="Handwritten dataset to download and train on",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size for train and validation loaders",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=4096,
        help="Maximum training examples to use for the bounded smoke run",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=1024,
        help="Maximum validation examples to use for the bounded smoke run",
    )
    parser.add_argument(
        "--sample-predictions",
        type=int,
        default=12,
        help="Number of prediction rows to attach to W&B at the end of training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for dataset sampling and model initialization",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Stormlog sampling interval in seconds",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Training device override",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default="wandb-character-cnn",
        help="Distributed job id/group label",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/wandb_character_cnn",
        help="Directory for local summary artifacts",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to cache downloaded dataset files",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="stormlog-smoke",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B entity/team",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline"],
        default="offline",
        help="W&B mode",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="wandb-character-cnn",
        help="W&B run name",
    )
    parser.add_argument(
        "--wandb-log-artifacts",
        action="store_true",
        help="Upload Stormlog output files and telemetry sink as W&B artifacts",
    )
    parser.add_argument(
        "--wandb-log-attribution",
        action="store_true",
        help="Log attribution HTML/tables when OOM dump artifacts are present",
    )
    return parser.parse_args()


def _resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but CUDA is not available.")
        return torch.device("cuda")
    if args.device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("Requested MPS device, but MPS is not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_torchvision() -> tuple[Any, Any]:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - exercised in runtime envs
        raise RuntimeError(
            "torchvision is required for this example. "
            "Install with `pip install torchvision`."
        ) from exc
    return datasets, transforms


def _dataset_root(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.data_dir:
        return Path(args.data_dir).resolve()
    return output_dir / "data"


def _subset_dataset(dataset: Any, limit: int, seed: int) -> Subset[Any]:
    limit = min(limit, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def _build_dataloaders(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    device: torch.device,
) -> tuple[DataLoader[Any], DataLoader[Any], list[str]]:
    datasets, transforms = _load_torchvision()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset_root = _dataset_root(args, output_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "kmnist":
        train_dataset = datasets.KMNIST(
            root=str(dataset_root),
            train=True,
            download=True,
            transform=transform,
        )
        val_dataset = datasets.KMNIST(
            root=str(dataset_root),
            train=False,
            download=True,
            transform=transform,
        )
        class_names = KMNIST_CLASS_NAMES
    else:
        train_dataset = datasets.MNIST(
            root=str(dataset_root),
            train=True,
            download=True,
            transform=transform,
        )
        val_dataset = datasets.MNIST(
            root=str(dataset_root),
            train=False,
            download=True,
            transform=transform,
        )
        class_names = [str(idx) for idx in range(10)]

    train_subset = _subset_dataset(train_dataset, args.train_samples, args.seed)
    val_subset = _subset_dataset(val_dataset, args.val_samples, args.seed + 1)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, class_names


def _build_tracker(
    args: argparse.Namespace,
    *,
    telemetry_sink_dir: Path,
    device: torch.device,
) -> Any:
    telemetry_sink_config = TelemetrySinkConfig(
        root_dir=telemetry_sink_dir,
        flush_every_seconds=1.0,
        rollover_max_bytes=32 * 1024 * 1024,
        retention_max_files=4,
        retention_max_total_bytes=128 * 1024 * 1024,
    )
    if device.type == "cpu":
        return CPUMemoryTracker(
            sampling_interval=args.interval,
            job_id=args.job_id,
            telemetry_sink_config=telemetry_sink_config,
        )
    tracker_device: str | None = "mps" if device.type == "mps" else None
    return MemoryTracker(
        device=tracker_device,
        sampling_interval=args.interval,
        enable_alerts=True,
        job_id=args.job_id,
        telemetry_sink_config=telemetry_sink_config,
    )


class CharacterCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_index: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0

    for step_index, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_loss = float(loss.detach().cpu().item())
        batch_accuracy = _accuracy(logits.detach(), targets.detach())
        total_loss += batch_loss
        total_accuracy += batch_accuracy

        global_step = epoch_index * len(loader) + step_index
        wandb.log(
            {
                "train/step_loss": batch_loss,
                "train/step_accuracy": batch_accuracy,
                "train/global_step": global_step,
            },
            step=global_step,
        )

    batch_count = max(1, len(loader))
    return {
        "loss": total_loss / batch_count,
        "accuracy": total_accuracy / batch_count,
    }


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        total_loss += float(criterion(logits, targets).detach().cpu().item())
        total_accuracy += _accuracy(logits, targets)

    batch_count = max(1, len(loader))
    return {
        "loss": total_loss / batch_count,
        "accuracy": total_accuracy / batch_count,
    }


@torch.no_grad()
def _log_prediction_table(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    class_names: list[str],
    row_limit: int,
) -> None:
    if row_limit <= 0:
        return

    model.eval()
    table = wandb.Table(
        columns=["index", "target", "prediction", "confidence", "image"],
    )
    added_rows = 0

    for inputs, targets in loader:
        logits = model(inputs.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu()
        predictions = probabilities.argmax(dim=1)

        for idx in range(inputs.size(0)):
            confidence = float(probabilities[idx, predictions[idx]].item())
            target_index = int(targets[idx].item())
            prediction_index = int(predictions[idx].item())
            image_tensor = inputs[idx].squeeze(0)
            image_pixels = (
                ((image_tensor + 1.0) * 127.5)
                .clamp(min=0, max=255)
                .to(torch.uint8)
                .numpy()
            )
            table.add_data(
                added_rows,
                class_names[target_index],
                class_names[prediction_index],
                confidence,
                wandb.Image(image_pixels),
            )
            added_rows += 1
            if added_rows >= row_limit:
                wandb.log({"evaluation/sample_predictions": table})
                return

    if added_rows > 0:
        wandb.log({"evaluation/sample_predictions": table})


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_sink_dir = output_dir / "telemetry_sink"
    telemetry_sink_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(args.seed)
    device = _resolve_device(args)
    train_loader, val_loader, class_names = _build_dataloaders(
        args,
        output_dir=output_dir,
        device=device,
    )
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device.type}")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.wandb_name,
        group=args.job_id,
        job_type="character-cnn-training",
        config={
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "sample_predictions": args.sample_predictions,
            "device_type": device.type,
        },
    )

    tracker = _build_tracker(
        args,
        telemetry_sink_dir=telemetry_sink_dir,
        device=device,
    )
    tracker.start_tracking()

    try:
        model = CharacterCNN(num_classes=len(class_names)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        history: list[dict[str, float]] = []

        for epoch_index in range(args.epochs):
            train_metrics = _train_one_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch_index=epoch_index,
            )
            val_metrics = _evaluate(
                model,
                val_loader,
                criterion=criterion,
                device=device,
            )
            epoch_summary = {
                "epoch": epoch_index + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            history.append(epoch_summary)
            wandb.log(
                {
                    "train/epoch_loss": train_metrics["loss"],
                    "train/epoch_accuracy": train_metrics["accuracy"],
                    "validation/loss": val_metrics["loss"],
                    "validation/accuracy": val_metrics["accuracy"],
                    "epoch": epoch_index + 1,
                },
                step=(epoch_index + 1) * len(train_loader),
            )
            print(
                f"Epoch {epoch_index + 1}/{args.epochs}:"
                f" train_loss={train_metrics['loss']:.4f}"
                f" train_acc={train_metrics['accuracy']:.4f}"
                f" val_loss={val_metrics['loss']:.4f}"
                f" val_acc={val_metrics['accuracy']:.4f}"
            )

        _log_prediction_table(
            model,
            val_loader,
            device=device,
            class_names=class_names,
            row_limit=args.sample_predictions,
        )
    finally:
        tracker.stop_tracking()

    stats = tracker.get_statistics()
    summary_payload = {
        "dataset": args.dataset,
        "device_type": device.type,
        "epochs": args.epochs,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "history": history,
        "stormlog_stats": stats,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    export_tracking_run_to_wandb(
        WandbExportConfig(
            enabled=True,
            log_artifacts=args.wandb_log_artifacts,
            log_attribution=args.wandb_log_attribution,
        ),
        command_name="wandb-character-cnn",
        session_summary=tracker.get_session_summary(),
        stats=stats,
        events=tracker.get_events(),
        output_path=summary_path,
        telemetry_sink_dir=telemetry_sink_dir,
        oom_dump_path=getattr(tracker, "last_oom_dump_path", None),
    )

    wandb.finish()
    final_epoch = history[-1] if history else None
    if final_epoch is not None:
        print(
            "Final metrics:"
            f" train_loss={final_epoch['train_loss']:.4f}"
            f" train_acc={final_epoch['train_accuracy']:.4f}"
            f" val_loss={final_epoch['val_loss']:.4f}"
            f" val_acc={final_epoch['val_accuracy']:.4f}"
        )
    print(f"Summary: {summary_path}")
    print(f"Telemetry sink: {telemetry_sink_dir}")


if __name__ == "__main__":
    main()
