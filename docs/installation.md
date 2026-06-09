[← Back to main docs](index.md)

# Installation Guide

Use the smallest install that matches your workflow, then validate the console scripts you plan to use.

## Choose an install profile

## Install name vs import names

Install the distribution as `stormlog`, then import the Python APIs from
`stormlog`, `stormlog.tensorflow`, or `stormlog.jax`:

| Task | Use |
| --- | --- |
| Install from PyPI | `pip install stormlog` |
| Launch the TUI | `stormlog` or `stormlog tui` |
| Profile OpenAI-compatible inference | `stormlog infer profile` |
| Import PyTorch APIs | `from stormlog import GPUMemoryProfiler, MemoryTracker` |
| Import TensorFlow APIs | `from stormlog.tensorflow import TFMemoryProfiler` |
| Import JAX APIs | `from stormlog.jax import JAXMemoryProfiler` |
| Run framework memory CLI automation | `gpumemprof`, `tfmemprof`, or `jaxmemprof` |

### Core package

```bash
pip install stormlog
```

Includes:

- `stormlog`
- `stormlog.tensorflow`
- `stormlog.jax`
- core telemetry and analysis utilities
- CPU-compatible monitoring and tracking

### Visualization extras

```bash
pip install "stormlog[viz]"
```

Adds the dependencies used by:

- `MemoryVisualizer`
- TUI HTML export from the Visualizations tab
- richer plot generation in example scripts

### TUI extras

```bash
pip install "stormlog[tui,torch]"
```

Installs the Textual stack plus the current PyTorch runtime dependency required by TUI startup. The `stormlog` console script is declared by the package; these extras make the app runnable.

The `stormlog` command remains TUI-first: running it with no arguments launches
the TUI. Use `stormlog query ...` for local artifact queries and
`stormlog infer ...` for OpenAI-compatible inference profiling.

### Inference tokenizer extras

```bash
pip install "stormlog[infer-tokenizers]"
```

Core `stormlog infer profile` uses the standard library HTTP stack and can run
without tokenizer packages. Install this extra when you want `tiktoken` or
`transformers` fallback token counts for endpoints that do not return usage
metadata.

### Framework extras

```bash
pip install "stormlog[torch]"
pip install "stormlog[tf]"
pip install "stormlog[jax]"
pip install "stormlog[all]"
```

`stormlog[all]` installs every runtime extra: `viz`, `tui`, `torch`, `tf`,
`jax`, `infer-tokenizers`, and W&B.

## Source checkout

```bash
git clone https://github.com/Silas-Asamoah/stormlog.git
cd stormlog
pip install -e .
```

For a contributor setup with all common extras:

```bash
pip install -e ".[dev,test,all]"
pre-commit install
```

### Source-only examples and guides

The `examples/` package and the Markdown test guides under
the [Example Test Guides](examples/test_guides/README.md) are only available
from a repository checkout. A plain `pip install stormlog` does not include them.

## Verification

### Core verification

```bash
python3 -c "import stormlog; print(stormlog.__version__)"
gpumemprof --help
gpumemprof info
```

### TensorFlow verification

If you installed the TensorFlow extra:

```bash
tfmemprof --help
tfmemprof info
```

### JAX verification

If you installed the JAX extra:

```bash
jaxmemprof --help
jaxmemprof info
```

### TUI verification

If you installed the TUI dependencies:

```bash
stormlog
stormlog tui
```

### Inference verification

```bash
stormlog infer --help
stormlog infer profile --help
```

If TUI dependencies are missing after a source checkout, reinstall with:

```bash
pip install -e ".[tui,torch]"
```

## Platform notes

### CPU-only systems

You do not need CUDA to use Stormlog. The CLI, TUI monitoring flows, and CPU profiler helpers remain usable on CPU-only machines. See the [CPU Compatibility Guide](cpu_compatibility.md).

### macOS and MPS

The TUI and CLI support Apple Silicon workflows. Use the CLI or TUI monitoring
and diagnostics flows even when `GPUMemoryProfiler` is unavailable on MPS.

### Linux and CUDA

Install the correct framework build for your `torch.cuda` runtime before using
`GPUMemoryProfiler`. For NVIDIA systems that means matching the CUDA toolchain.
See the [GPU Setup Guide](gpu_setup.md) for the full checklist.

## Common install failures

### `gpumemprof: command not found`

```bash
pip install -e .
hash -r
gpumemprof --help
```

### `stormlog: command not found`

```bash
pip install -e .
hash -r
stormlog --help
```

Install `stormlog[tui,torch]` as well if you want `stormlog` with no
arguments to launch the TUI successfully.

### Missing framework imports

Install the matching extra instead of trying to work around import errors manually:

```bash
pip install "stormlog[torch]"
pip install "stormlog[tf]"
pip install "stormlog[jax]"
```

## Next steps

1. Read the [Usage Guide](usage.md) for the Python API and CLI-first workflow.
2. Read the [CLI Guide](cli.md) if you need telemetry, plots, or diagnose bundles.
3. Read the [Inference Profiling Guide](inference.md) if you profile serving endpoints.
4. Read the [TUI Guide](tui.md) if you want an interactive terminal workflow.

---

[← Back to main docs](index.md)
