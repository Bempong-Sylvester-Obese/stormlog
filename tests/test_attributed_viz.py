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
