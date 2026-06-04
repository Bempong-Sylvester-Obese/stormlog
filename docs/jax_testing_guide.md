[← Back to main docs](index.md)

# JAX Testing Guide

This guide covers the current JAX workflow in Stormlog: profiling JAX code directly, tracking JAX memory usage from the CLI, and exporting artifacts for later review.

## Before you start

Validate the environment:

```bash
jaxmemprof info
```

If you are bringing up an accelerator runtime (CUDA or TPU), start with a basic JAX array operation before attempting complex tracking. These checks work on CPU-backed JAX installs as well.

## Daily workflow: ML engineer

Use `JAXMemoryProfiler` when you want snapshots and aggregate results around a real JAX workload.

```python
import jax.numpy as jnp
from stormlog.jax import JAXMemoryProfiler

profiler = JAXMemoryProfiler()

with profiler.profile_context("training"):
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    # JAX operations are asynchronous. Block until ready.
    y.block_until_ready()

results = profiler.get_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
print(f"Snapshots captured: {len(results.snapshots)}")
```

## Daily workflow: investigate sustained growth

The JAX CLI is the simplest way to capture longer-running telemetry:

```bash
jaxmemprof monitor --interval 0.5 --duration 30 --output jax_monitor.json
jaxmemprof track --interval 0.5 --output jax_track.json
jaxmemprof analyze --input jax_monitor.json --detect-leaks --optimize --report jax_report.txt
jaxmemprof diagnose --duration 0 --output ./jax_diag
```

For CPU-backed JAX or when the accelerator backend is unavailable, `jaxmemprof` will automatically fallback to CPU mode. You can explicitly force it with `--device cpu`.

## Recommended validation sequence

Use this when you need a compact JAX confidence pass:

```bash
jaxmemprof info
jaxmemprof monitor --interval 0.5 --duration 15 --output jax_monitor.json
jaxmemprof analyze --input jax_monitor.json --detect-leaks --optimize --report jax_report.txt
jaxmemprof diagnose --duration 0 --output ./jax_diag
```

## Common issues

### `jaxmemprof` runs on CPU when I expected GPU/TPU

Run:

```bash
jaxmemprof info
```

If the CLI outputs that JAX is running on CPU, you'll need to install the specific `jax` variants for your hardware (e.g., `jax[cuda12]`, `jax[tpu]`).

### Plot export fails

Install the visualization extra:

```bash
pip install "stormlog[viz]"
```

## Related docs

- [Usage Guide](usage.md)
- [CLI Guide](cli.md)
- [TUI Guide](tui.md)
- [Troubleshooting Guide](troubleshooting.md)

---

[← Back to main docs](index.md)
