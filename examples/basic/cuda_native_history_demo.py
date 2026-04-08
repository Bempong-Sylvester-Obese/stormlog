"""Generate CUDA allocator-history artifacts with the annotated Stormlog view."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from examples.common import (
    build_simple_torch_model,
    generate_torch_batch,
    get_torch_device,
    print_header,
    print_kv,
    print_section,
    seed_everything,
)
from stormlog.cuda_native_debug import (
    TRACE_HTML_ANNOTATED_FILENAME,
    capture_cuda_snapshot_artifacts,
    cuda_memory_history,
)


def _run_workload() -> dict[str, torch.Tensor]:
    model = build_simple_torch_model()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    retained: dict[str, torch.Tensor] = {}

    for step in range(3):
        inputs, targets = generate_torch_batch(batch_size=512)
        retained[f"batch_{step}"] = inputs[:128].detach().clone()

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        retained[f"logits_{step}"] = logits.detach().clone()
        loss.backward()
        optimizer.step()

    device = get_torch_device()
    retained["retained_projection"] = torch.nn.functional.gelu(
        torch.randn(4096, 1024, device=device)
    )
    retained["retained_scores"] = retained["retained_projection"] @ torch.randn(
        1024,
        128,
        device=device,
    )
    return retained


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small CUDA workload and write Stormlog allocator-history "
            "artifacts, including the annotated HTML timeline."
        )
    )
    parser.add_argument(
        "--output",
        default="./diag_bundle_native_demo",
        help="Output directory for the generated artifacts.",
    )
    parser.add_argument(
        "--trace-alloc-max-entries",
        type=int,
        default=100_000,
        help="Maximum allocator-history entries to retain while capturing.",
    )
    args = parser.parse_args()

    seed_everything()
    print_header("Stormlog - CUDA Native History Demo")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this example.")

    output_dir = Path(args.output).resolve()
    device = get_torch_device()

    print_section("Capture")
    print_kv("device", str(device))
    print_kv("output_dir", str(output_dir))

    with cuda_memory_history(
        device=device,
        trace_alloc_max_entries=args.trace_alloc_max_entries,
    ):
        retained = _run_workload()
        torch.cuda.synchronize()
        files_written = capture_cuda_snapshot_artifacts(
            output_dir,
            device=device,
            history_recorded=True,
        )

    print_section("Artifacts")
    print_kv("retained_tensors", len(retained))
    print_kv("files_written", len(files_written))
    print_kv(
        "annotated_html",
        str(output_dir / TRACE_HTML_ANNOTATED_FILENAME),
    )


if __name__ == "__main__":
    main()
