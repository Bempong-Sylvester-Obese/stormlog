from __future__ import annotations

import json
import re
from typing import Any, cast

import stormlog.attributed_viz as attributed_viz
import stormlog.cuda_native_debug as native_debug
from stormlog.jax.attributed_viz import render_jax_attributed_html


def _extract_embedded_payload(html: str) -> dict[str, Any]:
    prefix = "const DATA = "
    start = html.index(prefix) + len(prefix)
    end = html.index(";\n\n// === UTILS ===", start)
    return cast(dict[str, Any], json.loads(html[start:end].replace("<\\/", "</")))


def test_render_attributed_html_is_self_contained() -> None:
    html = attributed_viz.render_attributed_html(
        {"segments": [], "device_traces": [[]]},
        {"storage_pointer_count": 0, "attributed_storage_pointers": []},
    )

    assert "fonts.googleapis.com" not in html
    assert (
        native_debug.TRACE_HTML_ANNOTATED_FILENAME
        == attributed_viz.ATTRIBUTED_HTML_FILENAME
    )


def test_process_snapshot_falls_back_to_first_device_trace() -> None:
    snapshot = {
        "segments": [],
        "device_traces": [
            [
                {
                    "action": "alloc",
                    "addr": 8192,
                    "size": 64,
                    "time_us": 100,
                    "frames": [
                        {"name": "forward", "filename": "linear.py", "line": 12}
                    ],
                }
            ]
        ],
    }

    payload = (
        attributed_viz._process_snapshot(  # noqa: SLF001 - targeted regression coverage
            snapshot,
            {"storage_pointer_count": 0, "attributed_storage_pointers": []},
            device=1,
        )
    )

    assert payload["num_events"] == 1
    assert payload["events"][0]["action"] == "alloc"
    assert payload["events"][0]["name"] == "Activation (Linear)"


def test_render_attributed_html_escapes_script_closing_sequences() -> None:
    html = attributed_viz.render_attributed_html(
        {
            "segments": [],
            "device_traces": [
                [{"action": "alloc", "addr": 1, "size": 1, "time_us": 1}]
            ],
        },
        {
            "storage_pointer_count": 1,
            "attributed_storage_pointers": [
                {
                    "storage_ptr_int": 1,
                    "names": ["</script><script>alert(1)</script>"],
                    "tensors": [
                        {
                            "shape": [1],
                            "dtype": "torch.float32",
                            "size_bytes": 1,
                        }
                    ],
                }
            ],
        },
    )

    assert "</script><script>alert(1)</script>" not in html
    assert "<\\/script><script>alert(1)<\\/script>" in html


def test_process_snapshot_uses_block_address_for_active_allocations() -> None:
    snapshot = {
        "segments": [
            {
                "address": 4096,
                "segment_type": "large",
                "total_size": 128,
                "allocated_size": 32,
                "active_size": 32,
                "blocks": [
                    {
                        "address": 4096,
                        "size": 64,
                        "state": "inactive",
                        "frames": [],
                    },
                    {
                        "address": 12288,
                        "size": 32,
                        "state": "active_allocated",
                        "frames": [],
                    },
                ],
            }
        ],
        "device_traces": [[]],
    }
    tensor_index = {
        "storage_pointer_count": 1,
        "attributed_storage_pointers": [
            {
                "storage_ptr_int": 12288,
                "names": ["model.block.weight"],
                "tensors": [
                    {
                        "shape": [8, 8],
                        "dtype": "torch.float32",
                        "size_bytes": 32,
                    }
                ],
            }
        ],
    }

    payload = (
        attributed_viz._process_snapshot(  # noqa: SLF001 - targeted regression coverage
            snapshot,
            tensor_index,
        )
    )

    assert payload["active_table"][0]["address"] == 12288
    assert payload["active_table"][0]["name"] == "model.block.weight"


def test_process_snapshot_offenders_only_include_snapshot_active_allocations() -> None:
    snapshot = {
        "segments": [
            {
                "address": 8192,
                "segment_type": "large",
                "total_size": 128,
                "allocated_size": 32,
                "active_size": 32,
                "blocks": [
                    {
                        "address": 16384,
                        "size": 32,
                        "state": "active_allocated",
                        "frames": [],
                    }
                ],
            }
        ],
        "device_traces": [[]],
    }
    tensor_index = {
        "storage_pointer_count": 2,
        "attributed_storage_pointers": [
            {
                "storage_ptr_int": 16384,
                "names": ["active.tensor"],
                "tensors": [
                    {
                        "shape": [4, 4],
                        "dtype": "torch.float32",
                        "size_bytes": 32,
                    }
                ],
            },
            {
                "storage_ptr_int": 999999,
                "names": ["stale.live.tensor"],
                "tensors": [
                    {
                        "shape": [128, 128],
                        "dtype": "torch.float32",
                        "size_bytes": 65536,
                    }
                ],
            },
        ],
    }

    payload = (
        attributed_viz._process_snapshot(  # noqa: SLF001 - targeted regression coverage
            snapshot,
            tensor_index,
        )
    )

    assert [offender["name"] for offender in payload["offenders"]] == ["active.tensor"]


def test_process_snapshot_without_history_uses_active_snapshot_summary() -> None:
    snapshot = {
        "segments": [
            {
                "address": 4096,
                "segment_type": "large",
                "total_size": 256,
                "allocated_size": 128,
                "active_size": 128,
                "blocks": [
                    {
                        "address": 8192,
                        "size": 128,
                        "state": "active_allocated",
                        "frames": [],
                    }
                ],
            }
        ],
        "device_traces": [[]],
    }
    tensor_index = {
        "storage_pointer_count": 1,
        "attributed_storage_pointers": [
            {
                "storage_ptr_int": 8192,
                "names": ["model.linear.weight"],
                "tensors": [
                    {
                        "shape": [16, 8],
                        "dtype": "torch.float32",
                        "size_bytes": 128,
                    }
                ],
            }
        ],
    }

    payload = attributed_viz._process_snapshot(  # noqa: SLF001 - regression coverage
        snapshot,
        tensor_index,
    )

    assert payload["history_recorded"] is False
    assert payload["peak"] == 128
    assert payload["peak_label"] == "Active Alloc"
    assert payload["events_display"] == "n/a"


def test_render_attributed_html_embeds_timeline_segment_and_active_views() -> None:
    snapshot = {
        "segments": [
            {
                "address": 4096,
                "segment_type": "large",
                "total_size": 256,
                "allocated_size": 128,
                "active_size": 128,
                "blocks": [
                    {
                        "address": 8192,
                        "size": 128,
                        "state": "active_allocated",
                        "frames": [
                            {"name": "forward", "filename": "linear.py", "line": 12}
                        ],
                    },
                    {
                        "address": 12288,
                        "size": 128,
                        "state": "inactive",
                        "frames": [],
                    },
                ],
            }
        ],
        "device_traces": [
            [
                {
                    "action": "alloc",
                    "addr": 8192,
                    "size": 128,
                    "time_us": 100,
                    "frames": [
                        {"name": "forward", "filename": "linear.py", "line": 12}
                    ],
                }
            ]
        ],
    }
    tensor_index = {
        "storage_pointer_count": 1,
        "attributed_storage_pointers": [
            {
                "storage_ptr_int": 8192,
                "names": ["model.linear.weight"],
                "tensors": [
                    {
                        "shape": [16, 8],
                        "dtype": "torch.float32",
                        "size_bytes": 128,
                    }
                ],
            }
        ],
    }

    html = attributed_viz.render_attributed_html(snapshot, tensor_index)
    payload = _extract_embedded_payload(html)

    assert "Timeline Trace" in html
    assert "Segment Explorer" in html
    assert "Active Memory Table" in html
    assert "Top Memory Offenders" in html
    assert payload["num_events"] == 1
    assert payload["events"][0]["name"] == "model.linear.weight"
    assert payload["segments"][0]["blocks"][0]["name"] == "model.linear.weight"
    assert payload["active_table"][0]["name"] == "model.linear.weight"
    assert payload["offenders"][0]["name"] == "model.linear.weight"


def test_render_attributed_wandb_preview_html_is_static_and_sampled() -> None:
    traces = []
    for index in range(120):
        address = 8192 + index * 64
        traces.append(
            {
                "action": "alloc",
                "addr": address,
                "size": 64,
                "time_us": 100 + (index * 10),
                "frames": [
                    {"name": "forward", "filename": "linear.py", "line": index + 1}
                ],
            }
        )
        traces.append(
            {
                "action": "free_completed",
                "addr": address,
                "size": 64,
                "time_us": 105 + (index * 10),
                "frames": [],
            }
        )

    snapshot = {
        "segments": [
            {
                "address": 4096,
                "segment_type": "large",
                "total_size": 512,
                "allocated_size": 256,
                "active_size": 256,
                "blocks": [
                    {
                        "address": 16384,
                        "size": 256,
                        "state": "active_allocated",
                        "frames": [],
                    }
                ],
            }
        ],
        "device_traces": [traces],
    }
    tensor_index = {
        "storage_pointer_count": 1,
        "attributed_storage_pointers": [
            {
                "storage_ptr_int": 16384,
                "names": ["model.linear.weight"],
                "tensors": [
                    {
                        "shape": [32, 8],
                        "dtype": "torch.float32",
                        "size_bytes": 256,
                    }
                ],
            }
        ],
    }

    full_html = attributed_viz.render_attributed_html(snapshot, tensor_index)
    preview_html = attributed_viz.render_attributed_wandb_preview_html(
        snapshot,
        tensor_index,
        max_timeline_points=24,
        max_marker_points=8,
    )

    assert "Stormlog GPU Attribution Preview" in preview_html
    assert "Sampled W&amp;B preview" in preview_html
    assert "model.linear.weight" in preview_html
    assert "<script>" not in preview_html
    assert 'dominant-baseline="hanging"' in preview_html
    assert "preview-axis-band" in preview_html
    assert "stormlogPreviewPlotClip" in preview_html
    assert preview_html.index('fill="url(#stormlogPreviewArea)"') < preview_html.index(
        'class="preview-axis-band"'
    )
    assert len(preview_html) < len(full_html)


def test_sample_indices_handles_disabled_and_single_point_limits() -> None:
    assert attributed_viz._sample_indices(5, 0) == []  # noqa: SLF001
    assert attributed_viz._sample_indices(5, 1) == [4]  # noqa: SLF001


def test_render_attributed_wandb_preview_handles_one_point_sample_limit() -> None:
    snapshot = {
        "segments": [],
        "device_traces": [
            [
                {"action": "alloc", "addr": 8192, "size": 64, "time_us": 100},
                {"action": "alloc", "addr": 8256, "size": 64, "time_us": 200},
            ]
        ],
    }
    tensor_index = {"storage_pointer_count": 0, "attributed_storage_pointers": []}

    html = attributed_viz.render_attributed_wandb_preview_html(
        snapshot,
        tensor_index,
        max_timeline_points=1,
        max_marker_points=1,
    )

    assert "Stormlog GPU Attribution Preview" in html
    assert "1 plotted points" in html


def test_render_attributed_wandb_preview_labels_snapshot_only_view() -> None:
    snapshot = {
        "segments": [
            {
                "address": 4096,
                "segment_type": "large",
                "total_size": 128,
                "allocated_size": 64,
                "active_size": 64,
                "blocks": [
                    {
                        "address": 8192,
                        "size": 64,
                        "state": "active_allocated",
                        "frames": [],
                    }
                ],
            }
        ],
        "device_traces": [[]],
    }
    tensor_index = {
        "storage_pointer_count": 1,
        "attributed_storage_pointers": [
            {
                "storage_ptr_int": 8192,
                "names": ["model.linear.weight"],
                "tensors": [
                    {
                        "shape": [8, 8],
                        "dtype": "torch.float32",
                        "size_bytes": 64,
                    }
                ],
            }
        ],
    }

    html = attributed_viz.render_attributed_wandb_preview_html(snapshot, tensor_index)

    assert "Static W&amp;B preview from the live allocator snapshot" in html
    assert "recorded events" not in html
    assert "Trace Events" in html


def test_render_jax_attributed_html_keeps_root_to_leaf_order() -> None:
    html = render_jax_attributed_html(
        {"samples": [{"stack": ["root", "caller", "leaf"], "values": [1, 100]}]},
        output_path="",
    )

    leaf_row = re.search(
        r"<tr>\s*<td>\d+</td>\s*"
        r'<td style="[^"]*">leaf</td>\s*'
        r"<td>(?P<flat>.*?)</td>\s*"
        r"<td>(?P<cum>.*?)</td>",
        html,
        re.DOTALL,
    )
    root_row = re.search(
        r"<tr>\s*<td>\d+</td>\s*"
        r'<td style="[^"]*">root</td>\s*'
        r"<td>(?P<flat>.*?)</td>\s*"
        r"<td>(?P<cum>.*?)</td>",
        html,
        re.DOTALL,
    )

    assert leaf_row is not None
    assert root_row is not None
    assert "100.00B" in leaf_row.group("flat")
    assert "100.00B" in leaf_row.group("cum")
    assert "0.00B" in root_row.group("flat")
