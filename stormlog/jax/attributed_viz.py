"""Stormlog-native memory visualisation for JAX (Directed Graph Dashboard).

Generates a directed call-graph using Graphviz, identical to `go tool pprof`'s
graph view, and wraps it in a self-contained interactive HTML dashboard
with Top Allocations tables and Summary stats.
"""

import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def format_bytes(b: float) -> str:
    """Format bytes into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024.0:
            return f"{b:.2f}{unit}"
        b /= 1024.0
    return f"{b:.2f}PB"


def _compute_memory_stats(
    profile_data: Dict[str, Any],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[Tuple[str, str], int], int]:
    """Extract flat memory, cumulative memory, edges, and total memory from profile."""
    flat_mem: Dict[str, int] = {}
    cum_mem: Dict[str, int] = {}
    edges: Dict[Tuple[str, str], int] = {}
    total_mem = 0

    for sample in profile_data.get("samples", []):
        stack = list(sample["stack"])  # Root -> leaf, as normalized by pprof_parser.
        if not stack:
            continue

        bytes_val = (
            sample["values"][1] if len(sample["values"]) > 1 else sample["values"][0]
        )
        if bytes_val <= 0:
            continue

        total_mem += bytes_val
        leaf = stack[-1]

        flat_mem[leaf] = flat_mem.get(leaf, 0) + bytes_val

        # Deduplicate nodes in a single stack to avoid double-counting cumulative memory
        seen = set()
        for i, node in enumerate(stack):
            if node not in seen:
                cum_mem[node] = cum_mem.get(node, 0) + bytes_val
                seen.add(node)

            if i > 0:
                parent = stack[i - 1]
                edge = (parent, node)
                edges[edge] = edges.get(edge, 0) + bytes_val

    return flat_mem, cum_mem, edges, total_mem


def _generate_dot_graph(
    flat_mem: Dict[str, int],
    cum_mem: Dict[str, int],
    edges: Dict[Tuple[str, str], int],
    total_mem: int,
    threshold_pct: float = 0.01,
) -> str:
    """Generate Graphviz DOT source from the computed metrics."""
    if total_mem == 0:
        return 'digraph G { empty [label="No Memory Recorded"]; }'

    threshold_bytes = total_mem * threshold_pct

    # Filter nodes and edges by threshold
    valid_nodes = {n for n, c in cum_mem.items() if c >= threshold_bytes}

    dot = ["digraph G {"]
    dot.append('  node [shape=box, style=filled, fontname="Helvetica", fontsize=10];')
    dot.append('  edge [fontname="Helvetica", fontsize=9];')

    for node in valid_nodes:
        f_mem = flat_mem.get(node, 0)
        c_mem = cum_mem.get(node, 0)
        pct_c = (c_mem / total_mem) * 100

        # Color intensity based on Flat Memory (like pprof)
        # 0 flat = light gray/white, high flat = red
        intensity = min(1.0, f_mem / total_mem) if total_mem else 0
        r = 255
        g = int(255 * (1 - intensity))
        b = int(255 * (1 - intensity))
        color = f"#{r:02x}{g:02x}{b:02x}"

        node_safe = node.replace("<", "&lt;").replace(">", "&gt;")
        label = f"{node_safe}\\n{format_bytes(f_mem)} of {format_bytes(c_mem)} ({pct_c:.2f}%)"
        # Escape quotes
        node_id = node_safe.replace('"', '\\"').replace("\n", " ")
        dot.append(f'  "{node_id}" [label="{label}", fillcolor="{color}"];')

    for (parent, child), weight in edges.items():
        if parent in valid_nodes and child in valid_nodes and weight >= threshold_bytes:
            parent_safe = parent.replace("<", "&lt;").replace(">", "&gt;")
            child_safe = child.replace("<", "&lt;").replace(">", "&gt;")
            pid = parent_safe.replace('"', '\\"').replace("\n", " ")
            cid = child_safe.replace('"', '\\"').replace("\n", " ")

            # Edge thickness
            penwidth = max(1.0, min(5.0, (weight / total_mem) * 10))
            dot.append(
                f'  "{pid}" -> "{cid}" [label="{format_bytes(weight)}", penwidth={penwidth:.1f}];'
            )

    dot.append("}")
    return "\n".join(dot)


def render_jax_attributed_html(
    profile_data: Dict[str, Any], output_path: str = "jax_memory_graph.html"
) -> str:
    """Generate a self-contained HTML Dashboard from a parsed JAX pprof profile."""
    flat_mem, cum_mem, edges, total_mem = _compute_memory_stats(profile_data)
    dot_src = _generate_dot_graph(flat_mem, cum_mem, edges, total_mem)

    # Render SVG via Graphviz
    with tempfile.NamedTemporaryFile(suffix=".dot", mode="w", delete=False) as f:
        f.write(dot_src)
        dot_path = f.name

    try:
        svg_bytes = subprocess.check_output(["dot", "-Tsvg", dot_path])
        svg_content = svg_bytes.decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to run Graphviz 'dot': {e}")
        svg_content = """
        <div style="padding: 40px; text-align: center; color: #ff5555; background: #2d2d2d; border-radius: 8px;">
            <h2>Graphviz Required for Directed Graph</h2>
            <p>We attempted to generate the Directed Graph, but the <code>dot</code> command was not found on your system.</p>
            <p>Please install Graphviz to view the graphical flowchart:</p>
            <code style="background: #1e1e1e; padding: 10px; border-radius: 4px; display: inline-block; margin-top: 10px;">brew install graphviz</code> OR
            <code style="background: #1e1e1e; padding: 10px; border-radius: 4px; display: inline-block; margin-top: 10px;">sudo apt install graphviz</code>
            <br><br>
            <p><strong>Note:</strong> You can still use the <em>Top Allocations</em> tab to debug your memory usage perfectly without Graphviz!</p>
        </div>
        """
    finally:
        if os.path.exists(dot_path):
            os.unlink(dot_path)

    # Generate Top Allocations Table
    top_nodes = sorted(cum_mem.items(), key=lambda x: x[1], reverse=True)[:50]
    table_rows = ""
    for rank, (node, c_mem) in enumerate(top_nodes, 1):
        f_mem = flat_mem.get(node, 0)
        pct_c = (c_mem / total_mem) * 100 if total_mem else 0
        pct_f = (f_mem / total_mem) * 100 if total_mem else 0
        node_safe = node.replace("<", "&lt;").replace(">", "&gt;")
        table_rows += f"""
        <tr>
            <td>{rank}</td>
            <td style="font-family: monospace; text-align: left;">{node_safe}</td>
            <td>{format_bytes(f_mem)} <span style="color: #888; font-size: 0.85em;">({pct_f:.1f}%)</span></td>
            <td>{format_bytes(c_mem)} <span style="color: #888; font-size: 0.85em;">({pct_c:.1f}%)</span></td>
        </tr>
        """

    total_samples = len(profile_data.get("samples", []))

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Stormlog JAX OOM Dashboard</title>
    <style>
        body {{
            margin: 0; padding: 0; background: #1e1e1e; color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        .header {{
            background: #2d2d2d;
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #3d3d3d;
        }}
        .title {{ font-size: 1.2rem; font-weight: 600; color: #fff; }}
        .tabs {{
            display: flex;
            gap: 20px;
        }}
        .tab {{
            padding: 8px 16px;
            cursor: pointer;
            border-radius: 4px;
            font-weight: 500;
            color: #aaa;
            transition: all 0.2s;
        }}
        .tab:hover {{ background: #3d3d3d; color: #fff; }}
        .tab.active {{ background: #4a90e2; color: #fff; }}

        .content {{ display: none; padding: 20px; height: calc(100vh - 100px); overflow: auto; }}
        .content.active {{ display: block; }}

        #graph-content {{ display: flex; justify-content: center; align-items: flex-start; padding: 20px; box-sizing: border-box; }}
        svg {{ max-width: 100%; height: auto; background: white; border-radius: 4px; padding: 10px; }}

        table {{ width: 100%; border-collapse: collapse; background: #2d2d2d; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 15px; text-align: right; border-bottom: 1px solid #3d3d3d; }}
        th {{ background: #333; color: #ccc; font-weight: 600; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em; }}
        th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
        tr:hover {{ background: #353535; }}

        .summary-card {{ background: #2d2d2d; padding: 20px; border-radius: 8px; max-width: 600px; margin: 0 auto; }}
        .stat-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #3d3d3d; }}
        .stat-label {{ color: #aaa; }}
        .stat-value {{ font-weight: 600; font-size: 1.1rem; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">Stormlog JAX Memory Diagnostics</div>
        <div class="tabs">
            <div class="tab active" onclick="switchTab('graph', event)">Graph View</div>
            <div class="tab" onclick="switchTab('top', event)">Top Allocations</div>
            <div class="tab" onclick="switchTab('summary', event)">Summary</div>
        </div>
    </div>

    <!-- Graph View -->
    <div id="graph" class="content active" style="text-align: center;">
        {svg_content}
    </div>

    <!-- Top Allocations View -->
    <div id="top" class="content">
        <div style="max-width: 1200px; margin: 0 auto;">
            <h2 style="margin-top: 0;">Top 50 Memory Allocations</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Function Trace</th>
                        <th>Flat Memory (Self)</th>
                        <th>Cum. Memory (Total)</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
    </div>

    <!-- Summary View -->
    <div id="summary" class="content">
        <div class="summary-card">
            <h2 style="margin-top: 0; border-bottom: 2px solid #4a90e2; padding-bottom: 10px;">Diagnostic Summary</h2>
            <div class="stat-row">
                <span class="stat-label">Total Device Memory Tracked</span>
                <span class="stat-value" style="color: #ff5858;">{format_bytes(total_mem)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Allocation Samples</span>
                <span class="stat-value">{total_samples}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Unique Call Stack Nodes</span>
                <span class="stat-value">{len(cum_mem)}</span>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: #1e1e1e; border-left: 4px solid #4a90e2; border-radius: 0 4px 4px 0;">
                <p style="margin: 0; color: #aaa; font-size: 0.9rem; line-height: 1.5;">
                    <strong>Flat Memory</strong> is the memory allocated directly by the function itself.<br>
                    <strong>Cumulative Memory</strong> is the memory allocated by the function and all downstream functions it called.
                </p>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabId, event) {{
            // Hide all content
            document.querySelectorAll('.content').forEach(el => el.classList.remove('active'));
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));

            // Show selected content
            document.getElementById(tabId).classList.add('active');
            // Highlight selected tab
            if (event && event.target) {{
                event.target.classList.add('active');
            }}
        }}
    </script>
</body>
</html>"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(html)

    return html
