[← Back to main docs](index.md)

# GPU Stack Installation & CUDA Enablement

Use this guide to install the GPU-capable versions of PyTorch and TensorFlow,
verify CUDA availability, and switch between CPU and GPU smoke tests.

## 1. Prerequisites

- **NVIDIA driver** that supports the CUDA version you plan to use. Update
  through NVIDIA Experience (Windows) or your distro’s driver manager (Linux).
- **CUDA toolkit** (optional). Most pip wheels bundle the needed runtime, but
  native builds may require the toolkit/`nvcc`.
- **Python 3.10–3.12** in a virtual environment.

## 2. PyTorch (CUDA) Installation

Pick the CUDA build that matches your driver/toolkit. Example for CUDA 12.1:

```bash
# Windows / macOS / Linux (same pip command)
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

Need CUDA 11.8 instead? Swap the wheel URL:

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

### Verify PyTorch CUDA

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```

## 3. TensorFlow (GPU) Installation

TensorFlow ≥2.11 ships a unified wheel that includes GPU support as long as
the installed TensorFlow build matches the current driver/CUDA/cuDNN stack.
Install a TensorFlow version that is compatible with the host you are bringing
up. Example:

```bash
pip install tensorflow
```

### Verify TensorFlow CUDA

```bash
python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if not gpus:
    raise SystemExit("No GPU devices detected")
with tf.device("/GPU:0"):
    a = tf.random.normal((2048, 2048))
    b = tf.random.normal((2048, 2048))
    _ = tf.reduce_mean(tf.matmul(a, b)).numpy()
print("GPU matmul ok")
PY
```

## 4. Switching Between CPU and GPU Runs

- **CPU-only smoke test:** clear `CUDA_VISIBLE_DEVICES` and run:

  ```bash
  gpumemprof info
  gpumemprof track --duration 10 --interval 0.5 --output track.json --format json
  gpumemprof analyze track.json --format txt --output analysis.txt
  gpumemprof diagnose --duration 0 --output ./diag
  ```

  On Apple Silicon, clearing `CUDA_VISIBLE_DEVICES` disables CUDA but
  `gpumemprof info` may still report the `mps` backend. Treat this as a
  non-CUDA smoke test rather than a strict CPU-only force.

- **GPU path:** unset `CUDA_VISIBLE_DEVICES` (or set it to a GPU index) and run
  one known-good PyTorch workload plus the TensorFlow GPU matmul check:

  ```bash
  python -m examples.basic.pytorch_demo
  python - <<'PY'
  import tensorflow as tf
  from stormlog.tensorflow import TFMemoryProfiler

  profiler = TFMemoryProfiler(device="/GPU:0", enable_tensor_tracking=True)
  with profiler.profile_context("matmul_step"):
      a = tf.random.normal((4096, 4096))
      b = tf.random.normal((4096, 4096))
      c = tf.matmul(a, b)
      _ = tf.reduce_mean(c).numpy()

  results = profiler.get_results()
  print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
  print(f"Snapshots captured: {len(results.snapshots)}")
  PY
  ```

  After this path is clean, you can run `python -m examples.basic.tensorflow_demo`
  as the training-backed source-checkout example.

## 5. Common Issues

| Symptom | Fix |
| --- | --- |
| `torch.cuda.is_available()` is False | Confirm NVIDIA driver is installed, retry with the correct CUDA wheel, or reboot after driver install. |
| TensorFlow sees `/GPU:0` but training-backed ops fail | Confirm the GPU matmul check above succeeds first, then align the TensorFlow/CUDA/cuDNN stack before moving to Keras or cuDNN-backed demos. |
| TensorFlow cannot find cuDNN | Install a TensorFlow build that matches the current CUDA/cuDNN stack and rerun the GPU matmul check before using training-backed examples. |
| `RuntimeError: CUDA driver not found` | Check that `nvidia-smi` works on the command line; reinstall the driver if necessary. |
| CI path installing wrong framework | Follow `.github/workflows/ci.yml` logic: install the base deps, then exactly one framework (PyTorch or TensorFlow). |

## 6. Next Steps

Once the GPU stack is working, run the Textual TUI for an interactive check:

```bash
pip install "stormlog[tui,torch]"
stormlog
```

Use the overview tab to confirm the profiler sees your GPUs before running the
full benchmarking or release checklist.
