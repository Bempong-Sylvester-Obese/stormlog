"""JAX Memory Visualization"""

import logging
from typing import Any, Dict, Optional, Tuple

plt: Any
try:
    import matplotlib.pyplot as _plt

    plt = _plt
    MATPLOTLIB_AVAILABLE = True
    try:
        import seaborn as sns
    except ImportError:
        sns = None
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

jax: Any
try:
    import jax as _jax  # noqa: F401

    jax = _jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class MemoryVisualizer:
    """JAX memory visualization and dashboards."""

    def __init__(
        self, style: str = "default", figure_size: Tuple[int, int] = (12, 8)
    ) -> None:
        self.style = style
        self.figure_size = figure_size

        if MATPLOTLIB_AVAILABLE and style != "default":
            try:
                plt.style.use(style)
            except Exception:
                pass

    def plot_memory_timeline(
        self, results: Any, interactive: bool = False, save_path: Optional[str] = None
    ) -> None:
        """Plot device memory usage timeline."""
        if hasattr(results, "snapshots") and results.snapshots:
            timestamps = [s.timestamp for s in results.snapshots]
            memory_usage = [s.device_memory_mb for s in results.snapshots]
        elif hasattr(results, "memory_usage") and results.memory_usage:
            # Fallback for simple track results
            memory_usage = results.memory_usage
            timestamps = getattr(results, "timestamps", list(range(len(memory_usage))))
        else:
            logging.warning("No memory data available for plotting")
            return

        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=memory_usage,
                    mode="lines+markers",
                    name="Device Memory",
                    line=dict(color="crimson", width=2),
                )
            )

            fig.update_layout(
                title="Device Memory Usage Timeline",
                xaxis_title="Time",
                yaxis_title="Memory Usage (MB)",
                template="plotly_dark" if "dark" in self.style else "plotly",
            )

            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()

        elif MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=self.figure_size)
            plt.plot(
                timestamps,
                memory_usage,
                color="crimson",
                linewidth=2,
                label="Device Memory",
            )
            plt.title("Device Memory Usage Timeline")
            plt.xlabel("Time")
            plt.ylabel("Memory Usage (MB)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            else:
                plt.show()

    def plot_function_comparison(
        self,
        function_profiles: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot memory usage comparison for functions/contexts."""
        if not function_profiles:
            return

        functions = list(function_profiles.keys())
        peak_memories = [
            profile.get("peak_memory_bytes", 0) / (1024 * 1024)
            for profile in function_profiles.values()
        ]

        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=self.figure_size)
            plt.bar(functions, peak_memories, color="salmon", alpha=0.8)
            plt.title("Function Memory Comparison")
            plt.ylabel("Peak Memory (MB)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            else:
                plt.show()
