[← Back to Cookbook Index](index.md)

# JAX Production Recipes

This guide covers operational recipes for monitoring and troubleshooting JAX workloads with Stormlog.

## Profiling `jax.jit` functions

JAX uses XLA compilation under the hood, and caching is critical for performance and memory efficiency. You can profile `jax.jit` functions identically to standard JAX operations, and Stormlog will correctly track the underlying XLA allocations.

```python
import jax
import jax.numpy as jnp
from stormlog.jax import JAXMemoryProfiler

profiler = JAXMemoryProfiler()

@jax.jit
def fast_training_step(x):
    return jnp.dot(x, x)

with profiler.profile_context("jitted_step"):
    x = jnp.ones((1000, 1000))
    y = fast_training_step(x)
    y.block_until_ready()

results = profiler.get_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
```

## Wrapping functions for telemetry tracking

For complex architectures or library code where context managers are intrusive, you can use the `profile_function` decorator to instrument a JAX function globally.

```python
from stormlog.jax import profile_function
import jax.numpy as jnp

@profile_function(name="custom_matmul")
def custom_matmul(a, b):
    # This block will be transparently profiled
    res = jnp.dot(a, b)
    res.block_until_ready()
    return res
```

## Hardware and Device Placement

Stormlog correctly attributes memory tracking back to JAX devices. If you are operating on a multi-GPU/TPU setup and using `jax.sharding` or `jax.pmap`, Stormlog will aggregate memory profiles across the requested device scopes.

Ensure that the tracking target matches your runtime:
- **CUDA:** Requires `jax[cuda12]`
- **TPU:** Requires `jax[tpu]`
- **CPU:** Standard `jax` installation (used by `jaxmemprof` automatically if no accelerators are present)

## Advanced memory analytics

If you have exported a `jax_track.json` log using the CLI, you can pipe it into the Python API for offline heuristics (e.g. fragmentation checks or leak detection).

```python
from stormlog.jax.analyzer import MemoryAnalyzer
from stormlog.telemetry import TelemetryEventV2

# Assuming you loaded tracking events from a JSON log
events = [] # load JSON events

analyzer = MemoryAnalyzer()
findings = analyzer.analyze_memory_gaps(events)

for finding in findings:
    print(f"Gap detected: {finding.severity}")
```

---

[← Back to Cookbook Index](index.md)
