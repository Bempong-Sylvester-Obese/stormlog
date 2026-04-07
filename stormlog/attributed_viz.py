"""Stormlog-native memory visualisation with tensor attribution.

Generates a self-contained HTML page that renders an interactive CUDA memory
timeline where every allocation is labelled with its tensor name, shape, and
allocation site — replacing the opaque hex-address chart from PyTorch's
built-in ``_memory_viz.trace_plot()``.

Usage::

    from stormlog.attributed_viz import render_attributed_html

    html = render_attributed_html(snapshot_dict, tensor_index)
    Path("memory_attributed.html").write_text(html)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

ATTRIBUTED_HTML_FILENAME = "cuda_memory_attributed.html"


# ---------------------------------------------------------------------------
# Data processing — convert raw snapshot + attribution into a slim JSON
# ---------------------------------------------------------------------------


def _build_pointer_lookup(
    tensor_index: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    return {
        int(e["storage_ptr_int"]): e
        for e in tensor_index.get("attributed_storage_pointers", [])
    }


def _best_name(entry: Dict[str, Any]) -> str:
    names = entry.get("names", [])
    return names[0] if names else "<unnamed>"


def _shape_str(entry: Dict[str, Any]) -> str:
    tensors = entry.get("tensors", [])
    if not tensors:
        return ""
    shape = tensors[0].get("shape", [])
    return str(shape) if shape else ""


def _dtype_str(entry: Dict[str, Any]) -> str:
    tensors = entry.get("tensors", [])
    if not tensors:
        return ""
    return str(tensors[0].get("dtype", ""))


def _format_frames(frames: list) -> List[Dict[str, str]]:
    """Strip frames to lightweight dicts for JSON."""
    out = []
    for f in (frames or [])[:12]:
        out.append(
            {
                "name": f.get("name", ""),
                "file": f.get("filename", ""),
                "line": f.get("line", 0),
            }
        )
    return out


def _get_fallback_name(frames: List[Dict[str, str]]) -> str:
    """Extract a meaningful name for an unnamed tensor from its alloc stack."""
    for f in frames or []:
        name = f.get("name", "unknown")
        # Ignore completely generic wrapper frames
        if name in ("_call_impl", "_wrapped_call_impl", "<module>"):
            continue

        # If it's a generic 'forward', use the filename (e.g. linear.py -> Linear)
        if name == "forward":
            file = f.get("file", "")
            basename = file.split("/")[-1].replace(".py", "")
            if basename:
                # e.g. Activation (Linear), Activation (Activation)
                return f"Activation ({basename.capitalize()})"

        # For other functions (e.g. bmm, _in_projection_packed) strip leading _
        return f"Activation ({name.lstrip('_')})"

    return "Unnamed Tensor"


def _sz(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KiB"
    return f"{n} B"


def _process_snapshot(
    snapshot: Dict[str, Any],
    tensor_index: Dict[str, Any],
    device: int = 0,
) -> Dict[str, Any]:
    """Pre-process snapshot + attribution into a clean JSON payload."""
    ptr_map = _build_pointer_lookup(tensor_index)

    traces = snapshot.get("device_traces", [[]])[device]
    if not traces:
        traces = (
            snapshot.get("device_traces", [[]])[0]
            if snapshot.get("device_traces")
            else []
        )

    # Build timeline events
    events = []
    t0 = None
    cumulative = 0
    peak = 0
    live_allocs: Dict[int, Dict[str, Any]] = {}  # addr -> event info
    addr_to_frames = {}

    for t in traces:
        action = t.get("action", "")
        addr = t.get("addr", 0)
        size = t.get("size", 0)
        time_us = t.get("time_us", 0)

        if t0 is None:
            t0 = time_us

        # Only track alloc-level events for the chart
        if action == "alloc":
            cumulative += size
            peak = max(peak, cumulative)

            # Attribution lookup
            entry = ptr_map.get(addr)
            name = _best_name(entry) if entry else ""
            shape = _shape_str(entry) if entry else ""
            dtype = _dtype_str(entry) if entry else ""

            ev = {
                "t": round((time_us - t0) / 1000, 2),  # ms
                "action": "alloc",
                "addr": addr,
                "size": size,
                "size_h": _sz(size),
                "cum": cumulative,
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "frames": _format_frames(t.get("frames", [])),
            }
            addr_to_frames[addr] = ev["frames"]
            if not ev["name"] and ev["frames"]:
                ev["name"] = _get_fallback_name(ev["frames"])
            elif not ev["name"]:
                ev["name"] = "Unnamed Tensor"

            events.append(ev)
            live_allocs[addr] = ev

        elif action in ("free_requested", "free_completed"):
            if action == "free_completed":
                cumulative -= size
                live_allocs.pop(addr, None)
            events.append(
                {
                    "t": round((time_us - t0) / 1000, 2),
                    "action": action,
                    "addr": addr,
                    "size": size,
                    "size_h": _sz(size),
                    "cum": cumulative,
                    "name": "",
                    "shape": "",
                    "dtype": "",
                    "frames": _format_frames(t.get("frames", [])),
                }
            )

        elif action == "oom":
            events.append(
                {
                    "t": round((time_us - t0) / 1000, 2),
                    "action": "oom",
                    "addr": 0,
                    "size": size,
                    "size_h": _sz(size),
                    "cum": cumulative,
                    "name": "OOM",
                    "shape": "",
                    "dtype": "",
                    "frames": _format_frames(t.get("frames", [])),
                }
            )

    # Build "top offenders" from segments at OOM time
    raw_segments = snapshot.get("segments", [])
    offenders: List[Dict[str, Any]] = []

    for entry in tensor_index.get("attributed_storage_pointers", []):
        names = entry.get("names", [])
        tensors = entry.get("tensors", [])
        if not tensors:
            continue

        storage_ptr_int = int(entry.get("storage_ptr_int", 0))
        t0_tensor = tensors[0]

        name = names[0] if names else ""
        if not name:
            # Fallback to parsing allocation frame
            frames = addr_to_frames.get(storage_ptr_int, [])
            if frames:
                name = _get_fallback_name(frames)
            else:
                name = "Unnamed Tensor"

        offenders.append(
            {
                "name": name,
                "shape": str(t0_tensor.get("shape", [])),
                "dtype": t0_tensor.get("dtype", ""),
                "size": t0_tensor.get("size_bytes", 0),
                "size_h": _sz(t0_tensor.get("size_bytes", 0)),
                "count": len(tensors),
            }
        )
    offenders.sort(key=lambda x: x["size"], reverse=True)
    
    # Process Segments and embed block address/name lookups
    formatted_segments = []
    active_memory_table = []
    
    for s in raw_segments:
        formatted_blocks = []
        current_offset = s.get("address", 0)
        
        for b in s.get("blocks", []):
            block_size = b.get("size", 0)
            state = b.get("state", "")
            frames_fmt = _format_frames(b.get("frames", []))
            
            name = ""
            shape = ""
            dtype = ""
            
            if state == "active_allocated":
                entry = ptr_map.get(current_offset)
                if entry:
                    name = _best_name(entry)
                    shape = _shape_str(entry)
                    dtype = _dtype_str(entry)
                
                if not name and frames_fmt:
                    name = _get_fallback_name(frames_fmt)
                elif not name:
                    name = "Unnamed Tensor"
                    
                active_memory_table.append({
                    "address": current_offset,
                    "size": block_size,
                    "size_h": _sz(block_size),
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "frames": frames_fmt,
                    "pool": s.get("segment_type", "unknown")
                })
                
            formatted_blocks.append({
                "address": current_offset,
                "size": block_size,
                "size_h": _sz(block_size),
                "state": state,
                "name": name,
                "frames": frames_fmt
            })
            current_offset += block_size
            
        formatted_segments.append({
            "segment_type": s.get("segment_type", "large"),
            "address": s.get("address", 0),
            "total_size": s.get("total_size", 0),
            "total_size_h": _sz(s.get("total_size", 0)),
            "allocated_size": s.get("allocated_size", 0),
            "active_size": s.get("active_size", 0),
            "blocks": formatted_blocks,
        })
        
    active_memory_table.sort(key=lambda x: x["size"], reverse=True)

    # Summary of current segments
    total_reserved = sum(s.get("total_size", 0) for s in raw_segments)

    return {
        "events": events,
        "offenders": offenders[:50],
        "peak": peak,
        "peak_h": _sz(peak),
        "total_reserved": total_reserved,
        "total_reserved_h": _sz(total_reserved),
        "num_events": len(events),
        "num_segments": len(raw_segments),
        "attribution_count": tensor_index.get("storage_pointer_count", 0),
        "segments": formatted_segments,
        "active_table": active_memory_table
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stormlog — GPU Memory Attribution</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--surface:#161b22;--border:#30363d;
  --text:#e6edf3;--text2:#8b949e;--accent:#58a6ff;
  --red:#f85149;--green:#3fb950;--orange:#d29922;--purple:#bc8cff;
  --grad1:#58a6ff;--grad2:#bc8cff;
}
html,body{height:100%;background:var(--bg);color:var(--text);
  font-family:'Inter','SF Pro Display',-apple-system,system-ui,sans-serif;
  font-size:14px;line-height:1.5;overflow:hidden}
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Layout */
.app{display:grid;grid-template-rows:auto 1fr;grid-template-columns:1fr 380px;
  height:100vh;gap:0}
.header{grid-column:1/-1;padding:16px 24px;display:flex;align-items:center;
  gap:20px;border-bottom:1px solid var(--border);background:var(--surface)}
.header h1{font-size:18px;font-weight:700;
  background:linear-gradient(135deg,var(--grad1),var(--grad2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header .stats{display:flex;gap:16px;margin-left:auto}
.stat{text-align:center}
.stat .val{font-size:18px;font-weight:700;color:var(--accent);font-family:'JetBrains Mono',monospace}
.stat .lbl{font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px}

/* Chart area */
.chart-area{grid-column:1;grid-row:2;display:flex;flex-direction:column;
  overflow:hidden;border-right:1px solid var(--border);z-index:10}
.chart-container{flex:1;position:relative;padding:16px 24px 8px}
.chart-container svg{width:100%;height:100%}
.detail-panel{height:220px;border-top:1px solid var(--border);padding:12px 24px;
  overflow-y:auto;background:var(--surface);font-family:'JetBrains Mono',monospace;font-size:12px}
.detail-panel .detail-title{font-weight:600;color:var(--accent);margin-bottom:8px;font-size:14px;
  font-family:'Inter',sans-serif}
.detail-panel .detail-meta{color:var(--text2);margin-bottom:4px}
.detail-panel .detail-meta span{color:var(--text)}
.detail-panel .stack-frame{color:var(--text2);padding:1px 0}
.detail-panel .stack-frame .fn{color:var(--green)}
.detail-panel .stack-frame .loc{color:var(--text2)}

/* Right panel — offenders */
.right-panel{grid-column:2;grid-row:2;display:flex;flex-direction:column;
  overflow:hidden;background:var(--surface)}
.panel-header{padding:14px 20px;border-bottom:1px solid var(--border);
  font-weight:600;font-size:15px;color:var(--text)}
.offender-list{flex:1;overflow-y:auto;padding:8px 16px}
.offender{padding:10px 12px;border-radius:8px;margin-bottom:6px;cursor:pointer;
  border:1px solid transparent;transition:all .15s ease}
.offender:hover{border-color:var(--border);background:rgba(88,166,255,.06)}
.offender .off-header{display:flex;justify-content:space-between;align-items:baseline}
.offender .off-name{font-weight:600;font-size:13px;color:var(--text);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px}
.offender .off-size{font-weight:700;font-family:'JetBrains Mono',monospace;font-size:13px;
  white-space:nowrap}
.offender .off-detail{font-size:11px;color:var(--text2);margin-top:3px}
.offender .off-bar{height:4px;border-radius:2px;margin-top:6px;
  background:var(--border);overflow:hidden}
.offender .off-bar-fill{height:100%;border-radius:2px;
  background:linear-gradient(90deg,var(--grad1),var(--grad2))}
.off-rank{font-size:11px;color:var(--text2);font-weight:500;margin-right:6px;min-width:18px}

/* OOM marker */
.oom-line{stroke:var(--red);stroke-width:2;stroke-dasharray:6 3}
.oom-label{fill:var(--red);font-size:12px;font-weight:700;font-family:'Inter',sans-serif}

/* Tooltip */
.tooltip{position:absolute;pointer-events:none;background:rgba(22,27,34,.95);
  border:1px solid var(--border);border-radius:8px;padding:10px 14px;
  font-size:12px;font-family:'JetBrains Mono',monospace;max-width:400px;
  box-shadow:0 8px 24px rgba(0,0,0,.4);z-index:100;display:none}
.tooltip .tt-name{font-weight:600;color:var(--accent);font-size:13px;
  font-family:'Inter',sans-serif;margin-bottom:4px}
.tooltip .tt-info{color:var(--text2)}

/* Crosshair */
.crosshair{stroke:var(--text2);stroke-width:1;stroke-dasharray:3 3;opacity:.5}

/* Tabs */
.tabs-bar{grid-column:1/-1;display:flex;gap:12px;padding:0 24px;border-bottom:1px solid var(--border);background:var(--surface)}
.tab-btn{background:transparent;border:none;color:var(--text2);font-weight:600;font-size:13px;padding:12px 4px;cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}
.tab-btn:hover{color:var(--text)}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}

/* Tab Content */
.tab-content{display:none;grid-column:1/-1;grid-row:3;height:100%;overflow:hidden;flex-direction:column}
.tab-content.active{display:flex}
.app{grid-template-rows:auto auto 1fr;grid-template-columns:1fr;height:100vh}

/* Layout for Timeline Tab */
.timeline-layout{display:grid;grid-template-columns:1fr 380px;height:100%;width:100%;overflow:hidden}

/* Segments Tab */
.segments-pane{padding:24px;overflow-y:auto;flex:1}
.pool-title{font-weight:600;font-size:16px;color:var(--text);margin-bottom:12px;margin-top:24px;display:flex;align-items:center;gap:8px}
.segment-bar{display:flex;height:24px;border-radius:4px;overflow:hidden;margin-bottom:8px;background:rgba(255,255,255,0.05);border:1px solid var(--border)}
.seg-block{height:100%;cursor:pointer;transition:opacity .15s;min-width:1px}
.seg-block:hover{opacity:.8}
.seg-meta{font-size:12px;color:var(--text2);margin-bottom:4px;font-family:'JetBrains Mono',monospace;display:flex;justify-content:space-between}

/* Active Memory Table */
.active-pane{display:flex;flex-direction:column;height:100%;overflow:hidden}
.active-tbl-container{flex:1;overflow:auto;padding:0 24px 24px}
.active-tbl{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:12px}
.active-tbl th{position:sticky;top:0;background:var(--surface);padding:10px 8px;text-align:left;color:var(--text2);font-weight:600;border-bottom:1px solid var(--border);z-index:5}
.active-tbl td{padding:8px;border-bottom:1px solid var(--border)}
.active-tbl tbody tr:hover{background:rgba(88,166,255,.05)}
.active-tbl .t-name{font-family:'Inter',sans-serif;font-weight:600;color:var(--text)}
.active-badge{display:inline-block;padding:2px 6px;border-radius:4px;background:rgba(88,166,255,.15);color:var(--accent);font-size:10px}
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <h1>⚡ Stormlog GPU Attribution</h1>
    <div class="stats">
      <div class="stat"><div class="val" id="stat-peak">—</div><div class="lbl">Peak Alloc</div></div>
      <div class="stat"><div class="val" id="stat-events">—</div><div class="lbl">Events</div></div>
      <div class="stat"><div class="val" id="stat-tensors">—</div><div class="lbl">Tensors Tracked</div></div>
      <div class="stat"><div class="val" id="stat-segments">—</div><div class="lbl">Segments</div></div>
    </div>
  </div>
  <div class="tabs-bar">
    <button class="tab-btn active" onclick="switchTab('tab-timeline', this)">Timeline Trace</button>
    <button class="tab-btn" onclick="switchTab('tab-segments', this)">Segment Explorer</button>
    <button class="tab-btn" onclick="switchTab('tab-active', this)">Active Memory Table</button>
  </div>
  
  <div id="tab-timeline" class="tab-content active">
    <div class="timeline-layout">
      <div class="chart-area">
        <div class="chart-container" id="chart-container">
          <svg id="chart"></svg>
          <div class="tooltip" id="tooltip"></div>
        </div>
        <div class="detail-panel" id="detail-panel">
          <div class="detail-title">Click an allocation in the chart or offender list to inspect</div>
        </div>
      </div>
      <div class="right-panel">
        <div class="panel-header">🔥 Top Memory Offenders</div>
        <div class="offender-list" id="offender-list"></div>
      </div>
    </div>
  </div>

  <div id="tab-segments" class="tab-content">
    <div class="segments-pane" id="segments-pane"></div>
  </div>

  <div id="tab-active" class="tab-content">
    <div class="active-pane">
      <div style="padding:16px 24px;border-bottom:1px solid var(--border)">
        <input type="text" id="active-search" placeholder="Filter active allocations by name or dtype..." style="width:100%;padding:8px 12px;border-radius:6px;border:1px solid var(--border);background:#0d1117;color:#fff;font-family:'Inter',sans-serif">
      </div>
      <div class="active-tbl-container">
        <table class="active-tbl">
          <thead>
            <tr><th>Name</th><th>Size</th><th>Address</th><th>Shape</th><th>DType</th><th>Pool</th></tr>
          </thead>
          <tbody id="active-tbl-body"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
// === EMBEDDED DATA ===
const DATA = $DATA_JSON;

// === UTILS ===
function fmtSize(b){
  if(b>=1<<30)return(b/(1<<30)).toFixed(1)+' GiB';
  if(b>=1<<20)return(b/(1<<20)).toFixed(1)+' MiB';
  if(b>=1<<10)return(b/(1<<10)).toFixed(1)+' KiB';
  return b+' B';
}
function fmtTime(ms){
  if(ms>=1000)return(ms/1000).toFixed(2)+'s';
  return ms.toFixed(1)+'ms';
}

// === COLOR PALETTE ===
const COLORS = [
  '#58a6ff','#f0883e','#e15759','#3fb950','#bc8cff',
  '#edc949','#ff9da7','#76b7b2','#9c755f','#ff7b72',
  '#d2a8ff','#56d4dd','#ffa657','#79c0ff','#7ee787',
];
const nameColorMap = {};
let colorIdx = 0;
function colorFor(name){
  if(!name) return '#555';
  if(!nameColorMap[name]){
    nameColorMap[name] = COLORS[colorIdx % COLORS.length];
    colorIdx++;
  }
  return nameColorMap[name];
}

// === POPULATE STATS ===
document.getElementById('stat-peak').textContent = DATA.peak_h;
document.getElementById('stat-events').textContent = DATA.num_events.toLocaleString();
document.getElementById('stat-tensors').textContent = DATA.attribution_count;
document.getElementById('stat-segments').textContent = DATA.num_segments;

// === BUILD OFFENDER LIST ===
const offList = document.getElementById('offender-list');
const maxOffSize = DATA.offenders.length ? DATA.offenders[0].size : 1;
DATA.offenders.forEach((o, i) => {
  const pct = (o.size / maxOffSize * 100).toFixed(1);
  const color = colorFor(o.name);
  const el = document.createElement('div');
  el.className = 'offender';
  el.innerHTML = `
    <div class="off-header">
      <div style="display:flex;align-items:baseline;overflow:hidden">
        <span class="off-rank">#${i+1}</span>
        <span class="off-name" title="${o.name}">${o.name}</span>
      </div>
      <span class="off-size" style="color:${color}">${o.size_h}</span>
    </div>
    <div class="off-detail">${o.shape} · ${o.dtype}${o.count>1?' · '+o.count+' views':''}</div>
    <div class="off-bar"><div class="off-bar-fill" style="width:${pct}%;background:${color}"></div></div>
  `;
  el.addEventListener('click', () => showOffenderDetail(o, color));
  offList.appendChild(el);
});

function showOffenderDetail(o, color) {
  const dp = document.getElementById('detail-panel');
  dp.innerHTML = `
    <div class="detail-title" style="color:${color}">${o.name}</div>
    <div class="detail-meta">Shape: <span>${o.shape}</span></div>
    <div class="detail-meta">Dtype: <span>${o.dtype}</span></div>
    <div class="detail-meta">Size: <span>${o.size_h} (${o.size.toLocaleString()} bytes)</span></div>
    <div class="detail-meta">Views: <span>${o.count}</span></div>
  `;
}

// === CHART ===
const container = document.getElementById('chart-container');
const svg = document.getElementById('chart');
const tooltip = document.getElementById('tooltip');
const rect = container.getBoundingClientRect();
const W = rect.width - 48;
const H = rect.height - 24;

svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
svg.style.overflow = 'visible';

const events = DATA.events;
const allocEvents = events.filter(e => e.action === 'alloc');
const oomEvent = events.find(e => e.action === 'oom');

// Build cumulative timeline from ALL events (alloc + free_completed)
const timeline = [];
const relevantEvents = events.filter(e => e.action === 'alloc' || e.action === 'free_completed' || e.action === 'oom');
for (const e of relevantEvents) {
  timeline.push({ t: e.t, cum: e.cum, action: e.action, name: e.name, size: e.size, size_h: e.size_h });
}

if (timeline.length === 0) {
  svg.innerHTML = '<text x="50%" y="50%" text-anchor="middle" fill="#8b949e" font-size="16">No allocation events found</text>';
} else {

const tMin = timeline[0].t;
const tMax = timeline[timeline.length - 1].t;
const cumMax = Math.max(...timeline.map(d => d.cum)) * 1.05;

const margin = { top: 20, right: 20, bottom: 40, left: 72 };
const cw = W - margin.left - margin.right;
const ch = H - margin.top - margin.bottom;

function xScale(t) { return margin.left + (t - tMin) / (tMax - tMin || 1) * cw; }
function yScale(v) { return margin.top + ch - (v / cumMax) * ch; }

// Grid lines
const numYTicks = 6;
for (let i = 0; i <= numYTicks; i++) {
  const val = (cumMax / numYTicks) * i;
  const y = yScale(val);
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('x1', margin.left); line.setAttribute('x2', W - margin.right);
  line.setAttribute('y1', y); line.setAttribute('y2', y);
  line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', '1');
  svg.appendChild(line);
  const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  label.setAttribute('x', margin.left - 8); label.setAttribute('y', y + 4);
  label.setAttribute('text-anchor', 'end'); label.setAttribute('fill', '#8b949e');
  label.setAttribute('font-size', '11'); label.setAttribute('font-family', "'JetBrains Mono',monospace");
  label.textContent = fmtSize(val);
  svg.appendChild(label);
}

// X axis labels
const numXTicks = 8;
for (let i = 0; i <= numXTicks; i++) {
  const t = tMin + (tMax - tMin) / numXTicks * i;
  const x = xScale(t);
  const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  label.setAttribute('x', x); label.setAttribute('y', H - margin.bottom + 20);
  label.setAttribute('text-anchor', 'middle'); label.setAttribute('fill', '#8b949e');
  label.setAttribute('font-size', '11'); label.setAttribute('font-family', "'JetBrains Mono',monospace");
  label.textContent = fmtTime(t);
  svg.appendChild(label);
}

// Build area path
let pathD = `M ${xScale(timeline[0].t)} ${yScale(0)}`;
for (const p of timeline) {
  pathD += ` L ${xScale(p.t)} ${yScale(p.cum)}`;
}
pathD += ` L ${xScale(timeline[timeline.length-1].t)} ${yScale(0)} Z`;

// Gradient fill
const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
const grad = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
grad.setAttribute('id', 'areaGrad'); grad.setAttribute('x1', '0'); grad.setAttribute('y1', '0');
grad.setAttribute('x2', '0'); grad.setAttribute('y2', '1');
const s1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
s1.setAttribute('offset', '0%'); s1.setAttribute('stop-color', '#58a6ff'); s1.setAttribute('stop-opacity', '0.3');
const s2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
s2.setAttribute('offset', '100%'); s2.setAttribute('stop-color', '#58a6ff'); s2.setAttribute('stop-opacity', '0.02');
grad.appendChild(s1); grad.appendChild(s2); defs.appendChild(grad); svg.appendChild(defs);

const area = document.createElementNS('http://www.w3.org/2000/svg', 'path');
area.setAttribute('d', pathD); area.setAttribute('fill', 'url(#areaGrad)');
svg.appendChild(area);

// Line
let lineD = `M ${xScale(timeline[0].t)} ${yScale(timeline[0].cum)}`;
for (let i = 1; i < timeline.length; i++) {
  lineD += ` L ${xScale(timeline[i].t)} ${yScale(timeline[i].cum)}`;
}
const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
line.setAttribute('d', lineD); line.setAttribute('fill', 'none');
line.setAttribute('stroke', '#58a6ff'); line.setAttribute('stroke-width', '1.5');
svg.appendChild(line);

// Allocation dots — only named ones for clarity
const namedAllocs = allocEvents.filter(e => e.name);
// Sample if too many
const maxDots = 500;
const sampledAllocs = namedAllocs.length > maxDots
  ? namedAllocs.filter((_, i) => i % Math.ceil(namedAllocs.length / maxDots) === 0)
  : namedAllocs;

for (const e of sampledAllocs) {
  const cx = xScale(e.t);
  const cy = yScale(e.cum);
  const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  c.setAttribute('cx', cx); c.setAttribute('cy', cy); c.setAttribute('r', '3');
  c.setAttribute('fill', colorFor(e.name)); c.setAttribute('opacity', '0.8');
  c.setAttribute('stroke', 'none');
  c.style.cursor = 'pointer';
  c.addEventListener('mouseenter', (ev) => {
    c.setAttribute('r', '5'); c.setAttribute('opacity', '1');
    c.setAttribute('stroke', '#fff'); c.setAttribute('stroke-width', '1.5');
    showTooltip(ev, e);
  });
  c.addEventListener('mouseleave', () => {
    c.setAttribute('r', '3'); c.setAttribute('opacity', '0.8');
    c.setAttribute('stroke', 'none');
    hideTooltip();
  });
  c.addEventListener('click', () => showEventDetail(e));
  svg.appendChild(c);
}

// OOM marker
if (oomEvent) {
  const ox = xScale(oomEvent.t);
  const oomLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  oomLine.setAttribute('x1', ox); oomLine.setAttribute('x2', ox);
  oomLine.setAttribute('y1', margin.top); oomLine.setAttribute('y2', margin.top + ch);
  oomLine.setAttribute('class', 'oom-line');
  svg.appendChild(oomLine);
  const oomLbl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  oomLbl.setAttribute('x', ox - 4); oomLbl.setAttribute('y', margin.top - 4);
  oomLbl.setAttribute('text-anchor', 'end'); oomLbl.setAttribute('class', 'oom-label');
  oomLbl.textContent = '⚠ OOM — requested ' + oomEvent.size_h;
  svg.appendChild(oomLbl);
}

// Crosshair + interactive overlay
const crossV = document.createElementNS('http://www.w3.org/2000/svg', 'line');
crossV.setAttribute('class', 'crosshair'); crossV.style.display = 'none';
crossV.setAttribute('y1', margin.top); crossV.setAttribute('y2', margin.top + ch);
svg.appendChild(crossV);

const crossH = document.createElementNS('http://www.w3.org/2000/svg', 'line');
crossH.setAttribute('class', 'crosshair'); crossH.style.display = 'none';
crossH.setAttribute('x1', margin.left); crossH.setAttribute('x2', margin.left + cw);
svg.appendChild(crossH);

const overlay = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
overlay.setAttribute('x', margin.left); overlay.setAttribute('y', margin.top);
overlay.setAttribute('width', cw); overlay.setAttribute('height', ch);
overlay.setAttribute('fill', 'transparent'); overlay.style.cursor = 'crosshair';
svg.appendChild(overlay);

overlay.addEventListener('mousemove', (ev) => {
  const svgRect = svg.getBoundingClientRect();
  const scaleX = W / svgRect.width;
  const scaleY = H / svgRect.height;
  const mx = (ev.clientX - svgRect.left) * scaleX;
  const my = (ev.clientY - svgRect.top) * scaleY;
  crossV.setAttribute('x1', mx); crossV.setAttribute('x2', mx);
  crossH.setAttribute('y1', my); crossH.setAttribute('y2', my);
  crossV.style.display = ''; crossH.style.display = '';

  // Find nearest event
  const mouseT = tMin + (mx - margin.left) / cw * (tMax - tMin);
  let nearest = null, bestDist = Infinity;
  for (const e of timeline) {
    const d = Math.abs(e.t - mouseT);
    if (d < bestDist) { bestDist = d; nearest = e; }
  }
  if (nearest && nearest.name) {
    showTooltipAt(ev.clientX, ev.clientY, nearest);
  }
});
overlay.addEventListener('mouseleave', () => {
  crossV.style.display = 'none'; crossH.style.display = 'none';
  hideTooltip();
});

} // end if timeline.length

function showTooltip(ev, e) {
  tooltip.innerHTML = `
    <div class="tt-name" style="color:${colorFor(e.name)}">${e.name || 'unnamed'}</div>
    <div class="tt-info">${e.shape} · ${e.dtype}</div>
    <div class="tt-info">Alloc: ${e.size_h} · Cumulative: ${fmtSize(e.cum)}</div>
    <div class="tt-info">Time: ${fmtTime(e.t)}</div>
  `;
  tooltip.style.display = 'block';
  const cr = container.getBoundingClientRect();
  tooltip.style.left = (ev.clientX - cr.left + 12) + 'px';
  tooltip.style.top = (ev.clientY - cr.top - 40) + 'px';
}
function showTooltipAt(x, y, e) {
  tooltip.innerHTML = `
    <div class="tt-name">${e.name || e.action}</div>
    <div class="tt-info">Cumulative: ${fmtSize(e.cum)} · ${e.size_h}</div>
    <div class="tt-info">Time: ${fmtTime(e.t)}</div>
  `;
  tooltip.style.display = 'block';
  const cr = container.getBoundingClientRect();
  tooltip.style.left = (x - cr.left + 12) + 'px';
  tooltip.style.top = (y - cr.top - 40) + 'px';
}
function hideTooltip() { tooltip.style.display = 'none'; }

function showEventDetail(e) {
  const dp = document.getElementById('detail-panel');
  const framesHtml = e.frames.map(f => {
    const fn = f.name || '?';
    const file = f.file ? f.file.split('/').pop() : '';
    const loc = file ? `${file}:${f.line}` : '';
    return `<div class="stack-frame"><span class="fn">${fn}</span> <span class="loc">${loc}</span></div>`;
  }).join('');

  dp.innerHTML = `
    <div class="detail-title" style="color:${colorFor(e.name)}">${e.name || 'Unnamed Allocation'}</div>
    <div class="detail-meta">Shape: <span>${e.shape || '—'}</span> · Dtype: <span>${e.dtype || '—'}</span></div>
    <div class="detail-meta">Size: <span>${e.size_h} (${e.size.toLocaleString()} bytes)</span></div>
    <div class="detail-meta">Cumulative: <span>${fmtSize(e.cum)}</span> · Time: <span>${fmtTime(e.t)}</span></div>
    <div class="detail-meta">Address: <span>0x${e.addr.toString(16)}</span></div>
    <div style="margin-top:8px;border-top:1px solid var(--border);padding-top:8px">
      <div style="color:var(--text2);font-size:11px;margin-bottom:4px;font-family:Inter,sans-serif;font-weight:500">ALLOCATION STACK</div>
      ${framesHtml}
    </div>
  `;
}

// === TABS & UI ===
function switchTab(tabId, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(tabId).classList.add('active');
  if (tabId === 'tab-timeline') {
    svg.setAttribute('viewBox', `0 0 ${container.getBoundingClientRect().width - 48} ${container.getBoundingClientRect().height - 24}`);
  }
}

// === SEGMENTS EXPLORER ===
const segPane = document.getElementById('segments-pane');

function buildSegments() {
  const segments = DATA.segments || [];
  if (!segments.length) {
    segPane.innerHTML = '<div style="color:var(--text2)">No segment data available in snapshot.</div>';
    return;
  }
  
  const pools = { 'large': [], 'small': [], 'unknown': [] };
  segments.forEach(s => {
    if(pools[s.segment_type]) pools[s.segment_type].push(s);
    else pools['unknown'].push(s);
  });
  
  let html = '';
  Object.entries(pools).forEach(([type, segs]) => {
    if (!segs.length) return;
    
    html += `<div class="pool-title">
        <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:var(--accent)"></span>
        ${type.toUpperCase()} POOL
        <span class="active-badge">${segs.length} Segments</span>
      </div>`;
    
    segs.sort((a,b) => b.total_size - a.total_size).forEach(s => {
      const total = s.total_size;
      const allocated = s.allocated_size;
      const active = s.active_size;
      const pctAlloc = (allocated / total * 100).toFixed(1);
      
      html += `<div style="margin-bottom:24px">`;
      html += `<div class="seg-meta">
        <span><b>0x${s.address.toString(16)}</b> — ${s.total_size_h} Total</span>
        <span>${fmtSize(allocated)} Alloc (${pctAlloc}%) · ${fmtSize(active)} Active</span>
      </div>`;
      
      html += `<div class="segment-bar">`;
      s.blocks.forEach(b => {
        const pct = b.size / total * 100;
        let color = 'var(--border)'; // free
        if (b.state === 'active_allocated') color = colorFor(b.name);
        else if (b.state.includes('inactive')) color = '#d29922'; // yellow for fragmented
        
        const title = `${b.state} | ${b.size_h} | ${b.name || ''}`;
        html += `<div class="seg-block" style="width:${pct}%; background:${color}" title="${title}"></div>`;
      });
      html += `</div></div>`;
    });
  });
  segPane.innerHTML = html;
}

// === ACTIVE MEMORY TABLE ===
const tblBody = document.getElementById('active-tbl-body');
const searchInput = document.getElementById('active-search');

function renderActiveTable(query = '') {
  const activeTable = DATA.active_table || [];
  const q = query.toLowerCase();
  
  let html = '';
  const filtered = activeTable.filter(r => 
    !q || r.name.toLowerCase().includes(q) || (r.dtype && r.dtype.toLowerCase().includes(q))
  );
  
  filtered.forEach(r => {
    html += `<tr>
      <td><div class="t-name" style="color:${colorFor(r.name)}">${r.name}</div></td>
      <td>${r.size_h}</td>
      <td>0x${r.address.toString(16)}</td>
      <td>${r.shape || '—'}</td>
      <td>${r.dtype || '—'}</td>
      <td><span class="active-badge">${r.pool}</span></td>
    </tr>`;
  });
  tblBody.innerHTML = html;
}

if (searchInput) {
  searchInput.addEventListener('input', (e) => renderActiveTable(e.target.value));
}

// Init
buildSegments();
renderActiveTable();

</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_attributed_html(
    snapshot: Any,
    tensor_index: Dict[str, Any],
    *,
    device: int = 0,
) -> str:
    """Generate a self-contained HTML page with attributed memory timeline.

    Parameters
    ----------
    snapshot
        The raw dict from ``torch.cuda.memory._snapshot()`` or loaded from
        the pickle file.  Must have ``"segments"`` and ``"device_traces"``.
    tensor_index
        The dict from :func:`build_cuda_tensor_attribution_index`.
    device
        The CUDA device index to render.

    Returns
    -------
    str
        A complete HTML document string.
    """
    payload = _process_snapshot(snapshot, tensor_index, device=device)
    # Serialise with minimal whitespace
    data_json = json.dumps(payload, separators=(",", ":"))
    return _HTML_TEMPLATE.replace("$DATA_JSON", data_json)
