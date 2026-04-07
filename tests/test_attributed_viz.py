from __future__ import annotations

import stormlog.attributed_viz as attributed_viz
import stormlog.cuda_native_debug as native_debug


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
