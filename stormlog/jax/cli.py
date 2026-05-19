"""JAX Stormlog CLI"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

from stormlog.telemetry import telemetry_event_from_record, telemetry_event_to_dict
from stormlog.telemetry_sink import TelemetrySinkConfig
from stormlog.wandb_integration import (
    add_wandb_arguments,
    ensure_wandb_available,
    export_diagnose_bundle_to_wandb,
    export_tracking_run_to_wandb,
    wandb_config_from_namespace,
)

from .jax_env import configure_jax_logging
from .utils import format_memory, get_system_info

configure_jax_logging()

try:
    import jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None

if JAX_AVAILABLE:
    from .diagnose import run_diagnose
    from .tracker import JAXMemoryTracker


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _normalize_telemetry_events(
    records: list[dict[str, Any]], sampling_interval_ms: int
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        event = telemetry_event_from_record(
            record,
            default_collector="stormlog.jax.memory_tracker",
            default_sampling_interval_ms=sampling_interval_ms,
        )
        normalized.append(telemetry_event_to_dict(event))
    return normalized


def _build_telemetry_sink_config(
    args: argparse.Namespace,
) -> TelemetrySinkConfig | None:
    sink_dir = getattr(args, "telemetry_sink_dir", None)
    if not sink_dir:
        return None
    return TelemetrySinkConfig(
        root_dir=Path(sink_dir),
        flush_every_seconds=float(getattr(args, "telemetry_flush_seconds", 2.0)),
        rollover_max_bytes=int(getattr(args, "telemetry_rollover_mb", 64))
        * 1024
        * 1024,
        retention_max_files=int(getattr(args, "telemetry_retention_files", 8)),
        retention_max_total_bytes=int(
            getattr(args, "telemetry_retention_total_mb", 512)
        )
        * 1024
        * 1024,
    )


def _resolve_wandb_config(args: argparse.Namespace) -> Any:
    config = wandb_config_from_namespace(args)
    if not config.enabled:
        return config
    try:
        ensure_wandb_available(config)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None
    return config


def _warn_wandb_export_failure(command_name: str, exc: Exception) -> None:
    print(f"Warning: {command_name} W&B export skipped: {exc}", file=sys.stderr)


def cmd_info(args: argparse.Namespace) -> int:
    """Display system and device information."""
    print("JAX Stormlog - System Information")
    print("=" * 50)

    system_info = get_system_info()

    print(f"Platform: {system_info['platform']}")
    print(f"Python Version: {system_info['python_version']}")
    print(f"CPU Count: {system_info['cpu_count']}")

    if "total_memory_gb" in system_info:
        print(f"Total System Memory: {system_info['total_memory_gb']:.2f} GB")
        print(f"Available Memory: {system_info['available_memory_gb']:.2f} GB")

    backend_info = system_info.get("backend", {})
    device_count = backend_info.get("runtime_gpu_count", 0)

    print("\nJAX Backend Information:")
    print("-" * 30)
    print(f"Runtime Backend: {backend_info.get('runtime_backend', 'Unknown')}")
    print(f"Is XLA GPU Build: {backend_info.get('is_gpu_build', False)}")
    print(f"Is Apple Silicon: {backend_info.get('is_apple_silicon', False)}")
    print(f"JAX Metal Installed: {backend_info.get('jax_metal_installed', False)}")
    print(f"Available Devices: {device_count}")

    if device_count > 0:
        print("\nDevice Information:")
        print("-" * 20)
        from .utils import get_device_info

        for i in range(device_count):
            device_info = get_device_info(i)
            print(f"\nDevice {i}:")
            print(f"  Name: {device_info.get('device_kind', 'Unknown')}")
            stats = device_info.get("memory_stats", {})

            allocated_bytes = stats.get("bytes_in_use", 0)
            peak_bytes = stats.get("peak_bytes_in_use", 0)
            limit_bytes = stats.get("bytes_limit", 0)

            print(f"  Allocated Memory: {format_memory(allocated_bytes)}")
            print(f"  Peak Memory: {format_memory(peak_bytes)}")
            if limit_bytes:
                print(f"  Device Limit: {format_memory(limit_bytes)}")

    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    """Monitor device memory usage in real-time."""
    if not JAX_AVAILABLE:
        print("Error: JAX not available")
        return 1

    print("Starting JAX memory monitoring...")
    print(f"Sampling interval: {args.interval} seconds")
    print(
        f"Duration: {args.duration} seconds"
        if args.duration
        else "Duration: Indefinite"
    )
    if args.threshold:
        print(f"Alert threshold: {args.threshold} MB")
    print("Press Ctrl+C to stop\n")

    tracker = JAXMemoryTracker(
        sampling_interval=args.interval,
        alert_threshold_mb=args.threshold,
        device_index=args.device,
        enable_logging=True,
    )

    try:
        tracker.start_tracking()

        start_time = time.time()
        while True:
            if args.duration and (time.time() - start_time) >= args.duration:
                break

            current_memory = tracker.get_current_memory()
            print(
                f"\rCurrent memory usage: {current_memory:.1f} MB", end="", flush=True
            )

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\nStopping monitoring...")

    finally:
        results = tracker.stop_tracking()

        print("\nMonitoring Results:")
        print("-" * 20)
        print(f"Peak Memory: {results.peak_memory_bytes / (1024 * 1024):.1f} MB")
        print(f"Average Memory: {results.average_memory_bytes / (1024 * 1024):.1f} MB")
        print(f"Duration: {results.duration:.1f} seconds")
        print(f"Samples Collected: {len(results.memory_usage)}")
        dropped_samples = getattr(results, "history_dropped_samples", 0)
        if dropped_samples:
            print(f"Dropped Samples: {dropped_samples}")

        if results.alert_count:
            print(f"Alerts Triggered: {results.alert_count}")

        if args.output:
            # Save results
            output_data = {
                "peak_memory": results.peak_memory_bytes / (1024 * 1024),
                "average_memory": results.average_memory_bytes / (1024 * 1024),
                "duration": results.duration,
                "memory_usage": results.memory_usage,
                "timestamps": results.timestamps,
                "alerts": results.alert_count,
                "history_window_limit": getattr(results, "history_window_limit", 0),
                "history_retained_samples": getattr(
                    results, "history_retained_samples", 0
                ),
                "history_dropped_samples": getattr(
                    results, "history_dropped_samples", 0
                ),
            }

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)

            print(f"Results saved to {args.output}")

    return 0


def cmd_track(args: argparse.Namespace) -> int:
    """Start background memory tracking."""
    if not JAX_AVAILABLE:
        print("Error: JAX not available")
        return 1
    wandb_config = _resolve_wandb_config(args)
    if wandb_config is None:
        return 1

    print("Starting background memory tracking...")
    job_id = getattr(args, "job_id", None)
    rank = getattr(args, "rank", None)
    local_rank = getattr(args, "local_rank", None)
    world_size = getattr(args, "world_size", None)
    telemetry_sink_config = _build_telemetry_sink_config(args)

    tracker = JAXMemoryTracker(
        sampling_interval=args.interval,
        alert_threshold_mb=args.threshold,
        device_index=args.device,
        enable_logging=True,
        job_id=job_id,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        telemetry_sink_config=telemetry_sink_config,
        save_device_profile_on_stop=args.profile,
    )
    if telemetry_sink_config is not None:
        print(f"Append-only telemetry sink: {telemetry_sink_config.root_dir}")

    def alert_callback(alert: Dict[str, Any]) -> None:
        print(f"\n⚠️  MEMORY ALERT: {alert['message']}")

    tracker.add_alert_callback(alert_callback)

    try:
        tracker.start_tracking()
        print("Tracking started. Press Ctrl+C to stop and save results.")

        while True:
            time.sleep(5.0)

            stats = tracker.get_statistics()
            current_memory = stats.get("current_memory_mb")
            collector_health = str(stats.get("collector_health_status", "healthy"))
            if isinstance(current_memory, (int, float)):
                status_line = f"Current memory: {float(current_memory):.1f} MB"
            else:
                status_line = "Current memory: unavailable"
            status_line += f" | Health: {collector_health}"
            retry_at = stats.get("collector_next_retry_epoch_s")
            if isinstance(retry_at, (int, float)):
                retry_in = max(float(retry_at) - time.time(), 0.0)
                status_line += f" | Retry In: {retry_in:.1f}s"
            print(status_line)

    except KeyboardInterrupt:
        print("\nStopping tracking...")

    finally:
        results = tracker.stop_tracking()

        if args.output:
            sampling_interval_ms = int(round(args.interval * 1000))
            output_data = {
                "peak_memory": results.peak_memory_bytes / (1024 * 1024),
                "average_memory": results.average_memory_bytes / (1024 * 1024),
                "duration": results.duration,
                "memory_usage": results.memory_usage,
                "timestamps": results.timestamps,
                "alerts": results.alert_count,
                "events": _normalize_telemetry_events(
                    results.telemetry_events,
                    sampling_interval_ms=sampling_interval_ms,
                ),
                "history_window_limit": getattr(results, "history_window_limit", 0),
                "history_retained_samples": getattr(
                    results, "history_retained_samples", 0
                ),
                "history_dropped_samples": getattr(
                    results, "history_dropped_samples", 0
                ),
                "history_retained_events": getattr(
                    results, "history_retained_events", 0
                ),
                "history_dropped_events": getattr(results, "history_dropped_events", 0),
                "history_retained_alerts": getattr(
                    results, "history_retained_alerts", 0
                ),
                "history_dropped_alerts": getattr(results, "history_dropped_alerts", 0),
            }

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)

            print(f"Results saved to {args.output}")

        final_stats = tracker.get_statistics()
        print(
            f"\nTracking completed. Peak memory: {results.peak_memory_bytes / (1024 * 1024):.1f} MB"
        )
        dropped_samples = int(final_stats.get("history_dropped_samples", 0))
        dropped_events = int(final_stats.get("history_dropped_events", 0))
        if dropped_samples or dropped_events:
            print(
                "History truncation: "
                f"samples={dropped_samples}, events={dropped_events}"
            )
        if "collector_health_status" in final_stats:
            print(
                "Collector health: "
                f"{final_stats.get('collector_health_status', 'healthy')}"
            )
        if final_stats.get("collector_last_error"):
            print(f"Last collector error: {final_stats.get('collector_last_error')}")

        if results.device_memory_profile_path:
            print(
                f"Device memory profile saved to: {results.device_memory_profile_path}"
            )

        if wandb_config.enabled:
            try:
                export_tracking_run_to_wandb(
                    wandb_config,
                    command_name="jaxmemprof-track",
                    session_summary=tracker.get_session_summary(),
                    stats=final_stats,
                    events=results.telemetry_events,
                    output_path=args.output,
                    telemetry_sink_dir=getattr(args, "telemetry_sink_dir", None),
                    oom_dump_path=results.device_memory_profile_path,
                )
                print("W&B export completed.")
            except Exception as exc:
                _warn_wandb_export_failure("jaxmemprof track", exc)

    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Produce a portable diagnostic bundle. Returns 0 (OK), 1 (failure), or 2 (memory risk)."""
    if args.duration < 0:
        print("Error: --duration must be >= 0", file=sys.stderr)
        return 1
    if args.interval <= 0:
        print("Error: --interval must be > 0", file=sys.stderr)
        return 1

    wandb_config = _resolve_wandb_config(args)
    if wandb_config is None:
        return 1

    command_line = " ".join(sys.argv)
    try:
        artifact_dir, exit_code = run_diagnose(
            output=args.output,
            device_index=args.device,
            duration=args.duration,
            interval=args.interval,
            command_line=command_line,
        )
    except OSError:
        return 1

    # Structured stdout summary
    print(f"Artifact: {artifact_dir}")
    if exit_code == 0:
        status = "OK"
    elif exit_code == 2:
        status = "MEMORY_RISK"
    else:
        status = "FAILED"
    print(f"Status: {status} (exit_code={exit_code})")

    try:
        manifest_path = artifact_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest.get("risk_detected"):
                summary_path = artifact_dir / "diagnostic_summary.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        summary = json.load(f)
                    flags = summary.get("risk_flags", {})
                    parts = [k for k, v in flags.items() if v]
                    if parts:
                        print(f"Findings: {', '.join(parts)}")
        if exit_code == 0 and status == "OK":
            print("Findings: no memory risk detected")
    except (OSError, json.JSONDecodeError):
        pass

    if wandb_config.enabled:
        try:
            export_diagnose_bundle_to_wandb(
                wandb_config,
                command_name="jaxmemprof-diagnose",
                artifact_dir=artifact_dir,
            )
            print("W&B export completed.")
        except Exception as exc:
            _warn_wandb_export_failure("jaxmemprof diagnose", exc)

    return exit_code


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="JAX Stormlog CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cookbook:
  https://stormlog.readthedocs.io/en/latest/cookbook/index.html
        """,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    subparsers.add_parser("info", help="Display system and device information")

    # Monitor command
    monitor_parser = subparsers.add_parser(
        "monitor", help="Monitor device memory usage"
    )
    monitor_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    monitor_parser.add_argument(
        "--duration",
        type=float,
        help="Monitoring duration in seconds (default: indefinite)",
    )
    monitor_parser.add_argument(
        "--threshold", type=float, help="Memory alert threshold in MB"
    )
    monitor_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="JAX device index to monitor (default: 0)",
    )
    monitor_parser.add_argument("--output", help="Output file for results")

    # Track command
    track_parser = subparsers.add_parser("track", help="Background memory tracking")
    track_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    track_parser.add_argument(
        "--threshold",
        type=float,
        default=4000,
        help="Memory alert threshold in MB (default: 4000)",
    )
    track_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="JAX device index to monitor (default: 0)",
    )
    track_parser.add_argument(
        "--profile",
        action="store_true",
        help="Export a JAX device memory profile upon tracking completion",
    )
    track_parser.add_argument(
        "--job-id",
        default=None,
        help="Distributed job identifier override (default: infer from env)",
    )
    track_parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global distributed rank override (default: infer from env)",
    )
    track_parser.add_argument(
        "--local-rank",
        type=int,
        default=None,
        help="Local distributed rank override (default: infer from env)",
    )
    track_parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Distributed world size override (default: infer from env)",
    )
    track_parser.add_argument(
        "--output", required=True, help="Output file for tracking results"
    )
    track_parser.add_argument(
        "--telemetry-sink-dir",
        default=None,
        help="Directory for append-only telemetry sink segments",
    )
    track_parser.add_argument(
        "--telemetry-flush-seconds",
        type=float,
        default=2.0,
        help="Maximum seconds between telemetry sink flushes (default: 2.0)",
    )
    track_parser.add_argument(
        "--telemetry-rollover-mb",
        type=int,
        default=64,
        help="Telemetry sink segment rollover size in MB (default: 64)",
    )
    track_parser.add_argument(
        "--telemetry-retention-files",
        type=int,
        default=8,
        help="Maximum retained telemetry sink segments (default: 8)",
    )
    track_parser.add_argument(
        "--telemetry-retention-total-mb",
        type=int,
        default=512,
        help="Maximum retained telemetry sink size in MB (default: 512)",
    )
    add_wandb_arguments(track_parser)

    # Diagnose command
    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Produce a portable diagnostic bundle for debugging memory failures",
    )
    diagnose_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for the artifact bundle (default: cwd)",
    )
    diagnose_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="JAX device index to monitor (default: 0)",
    )
    diagnose_parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Seconds to run tracker for telemetry (default: 5, use 0 to skip)",
    )
    diagnose_parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Sampling interval for timeline (default: 0.5)",
    )
    add_wandb_arguments(diagnose_parser)

    args = parser.parse_args()

    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    if args.command == "info":
        return cmd_info(args)
    elif args.command == "monitor":
        return cmd_monitor(args)
    elif args.command == "track":
        return cmd_track(args)
    elif args.command == "diagnose":
        return cmd_diagnose(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
