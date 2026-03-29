"""
Real-time TensorFlow Memory Tracking

This module provides real-time monitoring of GPU memory usage during TensorFlow
model training and inference, with configurable alerts and automatic cleanup.
"""

import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .tf_env import configure_tensorflow_logging

configure_tensorflow_logging()

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from stormlog.collector_health import (
    COLLECTOR_HEALTH_HEALTHY,
    COLLECTOR_HEALTH_UNHEALTHY,
    CollectorHealthState,
    collector_retry_delay_seconds,
)
from stormlog.telemetry import (
    resolve_distributed_identity,
    telemetry_event_from_record,
    telemetry_event_to_dict,
)


@dataclass
class TrackingResult:
    """Results from real-time memory tracking."""

    start_time: float
    end_time: float
    memory_usage: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)
    alerts_triggered: List[Dict] = field(default_factory=list)
    peak_memory: float = 0.0
    average_memory: float = 0.0
    min_memory: float = float("inf")

    @property
    def duration(self) -> float:
        """Total tracking duration."""
        return self.end_time - self.start_time

    @property
    def memory_growth_rate(self) -> float:
        """Memory growth rate in MB/second."""
        if len(self.memory_usage) < 2 or self.duration <= 0:
            return 0.0
        return (self.memory_usage[-1] - self.memory_usage[0]) / self.duration


class MemoryTracker:
    """Real-time TensorFlow GPU memory tracker."""

    def __init__(
        self,
        sampling_interval: float = 1.0,
        alert_threshold_mb: Optional[float] = None,
        device: Optional[str] = None,
        enable_logging: bool = True,
        job_id: Optional[str] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        """
        Initialize memory tracker.

        Args:
            sampling_interval: Time between memory samples in seconds
            alert_threshold_mb: Memory threshold for alerts in MB
            device: TensorFlow device to monitor (e.g., '/GPU:0')
            enable_logging: Whether to log events
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available. Please install TensorFlow.")
        if sampling_interval <= 0:
            raise ValueError("sampling_interval must be > 0")

        self.sampling_interval = sampling_interval
        self.alert_threshold_mb = alert_threshold_mb
        self.device = device or self._get_default_device()
        self.enable_logging = enable_logging
        self.distributed_identity = resolve_distributed_identity(
            job_id=job_id,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            env=os.environ,
        )

        # Tracking state
        self.tracking = False
        self.tracking_thread: Optional[threading.Thread] = None
        self.memory_usage: List[float] = []
        self.timestamps: List[float] = []
        self.events: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []

        # Thread synchronization
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._collector_health = CollectorHealthState()
        self._last_successful_memory_mb: Optional[float] = None
        self._session_start_time: Optional[float] = None
        self._session_end_time: Optional[float] = None
        self._collector_retry_backoff_initial_s = 1.0
        self._collector_retry_backoff_factor = 2.0
        self._collector_retry_backoff_cap_s = 30.0

        # Alert callbacks
        self.alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        if enable_logging:
            logging.info(f"TensorFlow Memory Tracker initialized for {self.device}")

    def _device_id(self) -> int:
        """Best-effort device id extraction."""
        if isinstance(self.device, str):
            if "CPU" in self.device.upper():
                return -1
            if ":" in self.device:
                tail = self.device.rsplit(":", 1)[-1]
                if tail.isdigit():
                    return int(tail)
            if "/GPU" in self.device.upper():
                return 0
        return -1

    def _build_telemetry_event_record(
        self,
        *,
        timestamp: float,
        memory_mb: float,
        event_type: str = "sample",
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sampling_interval_ms = int(round(self.sampling_interval * 1000))
        legacy = {
            "timestamp": timestamp,
            "type": event_type,
            "memory_mb": memory_mb,
            "device_id": self._device_id(),
            "context": context,
            "metadata": {
                **dict(metadata or {}),
                **self._collector_health.to_dict(),
            },
            "collector": "stormlog.tensorflow.memory_tracker",
            "sampling_interval_ms": sampling_interval_ms,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "job_id": self.distributed_identity["job_id"],
            "rank": self.distributed_identity["rank"],
            "local_rank": self.distributed_identity["local_rank"],
            "world_size": self.distributed_identity["world_size"],
        }
        event = telemetry_event_from_record(
            legacy,
            default_collector="stormlog.tensorflow.memory_tracker",
            default_sampling_interval_ms=sampling_interval_ms,
        )
        return telemetry_event_to_dict(event)

    def _get_default_device(self) -> str:
        """Get default TensorFlow device."""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                return "/GPU:0"
            else:
                return "/CPU:0"
        except Exception as exc:
            logging.debug("Default device detection failed: %s", exc)
            return "/CPU:0"

    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        if "/GPU:" in self.device:
            # Extract GPU index from device string
            gpu_id = int(self.device.split(":")[1]) if ":" in self.device else 0
            memory_info = tf.config.experimental.get_memory_info(f"/GPU:{gpu_id}")
            current_bytes = memory_info.get("current", 0)
            if isinstance(current_bytes, (int, float)):
                return float(current_bytes) / (1024 * 1024)
            raise RuntimeError("TensorFlow memory info returned a non-numeric value")

        # CPU memory tracking
        import psutil

        process = psutil.Process()
        return float(process.memory_info().rss) / (1024 * 1024)

    def _set_collector_health(
        self,
        *,
        status: str,
        telemetry_partial: bool,
        last_error: Optional[str] = None,
        consecutive_failures: int = 0,
        next_retry_epoch_s: Optional[float] = None,
    ) -> None:
        self._collector_health = CollectorHealthState(
            status=status,
            telemetry_partial=telemetry_partial,
            last_error=last_error,
            consecutive_failures=consecutive_failures,
            next_retry_epoch_s=next_retry_epoch_s,
        )

    def _retry_collection_due(self, now: float) -> bool:
        retry_at = self._collector_health.next_retry_epoch_s
        return retry_at is None or now >= retry_at

    def _status_memory_value(self) -> float:
        return float(self._last_successful_memory_mb or 0.0)

    def _append_event(
        self,
        *,
        timestamp: float,
        memory_mb: float,
        event_type: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self.events.append(
                self._build_telemetry_event_record(
                    timestamp=timestamp,
                    memory_mb=memory_mb,
                    event_type=event_type,
                    context=context,
                    metadata=metadata,
                )
            )

    def _transition_to_failure(self, timestamp: float, exc: BaseException) -> None:
        previous_health = self._collector_health
        consecutive_failures = previous_health.consecutive_failures + 1
        retry_delay_s = collector_retry_delay_seconds(
            consecutive_failures,
            initial_delay_s=self._collector_retry_backoff_initial_s,
            factor=self._collector_retry_backoff_factor,
            max_delay_s=self._collector_retry_backoff_cap_s,
        )
        next_retry_epoch_s = timestamp + retry_delay_s if retry_delay_s > 0 else None
        error_message = str(exc)
        self._set_collector_health(
            status=COLLECTOR_HEALTH_UNHEALTHY,
            telemetry_partial=True,
            last_error=error_message,
            consecutive_failures=consecutive_failures,
            next_retry_epoch_s=next_retry_epoch_s,
        )
        if previous_health.status == COLLECTOR_HEALTH_HEALTHY:
            self._append_event(
                timestamp=timestamp,
                memory_mb=self._status_memory_value(),
                event_type="collector_degraded",
                context="Collector unavailable; telemetry paused until recovery.",
                metadata={
                    "collector_transition": "degraded",
                    "collector_degraded_from": previous_health.status,
                    "collector_degradation_reason": error_message,
                    "collector_retry_delay_s": retry_delay_s,
                },
            )
        if self.enable_logging:
            logging.warning("Could not get memory usage: %s", error_message)

    def _transition_to_success(self, timestamp: float) -> None:
        previous_health = self._collector_health
        previous_error = previous_health.last_error
        previous_failures = previous_health.consecutive_failures
        if previous_health.status != COLLECTOR_HEALTH_HEALTHY:
            self._set_collector_health(
                status=COLLECTOR_HEALTH_HEALTHY,
                telemetry_partial=False,
            )
            self._append_event(
                timestamp=timestamp,
                memory_mb=self._status_memory_value(),
                event_type="collector_recovered",
                context="Collector recovered; full telemetry sampling resumed.",
                metadata={
                    "collector_transition": "recovered",
                    "collector_recovered_from": previous_health.status,
                    "collector_previous_error": previous_error,
                    "collector_previous_failure_count": previous_failures,
                },
            )
            return
        self._set_collector_health(
            status=COLLECTOR_HEALTH_HEALTHY,
            telemetry_partial=False,
        )

    def _run_tracking_iteration(self) -> None:
        """Collect one tracking sample or advance degraded-mode state."""
        current_time = time.time()
        if not self._retry_collection_due(current_time):
            return

        try:
            current_memory = self._get_current_memory()
        except Exception as exc:
            self._transition_to_failure(current_time, exc)
            return

        self._last_successful_memory_mb = current_memory
        self._transition_to_success(current_time)

        with self._lock:
            self.memory_usage.append(current_memory)
            self.timestamps.append(current_time)
            self.events.append(
                self._build_telemetry_event_record(
                    timestamp=current_time,
                    memory_mb=current_memory,
                    event_type="sample",
                )
            )

        if self.alert_threshold_mb and current_memory > self.alert_threshold_mb:
            self._trigger_alert(current_memory, current_time)

    def _tracking_loop(self) -> None:
        """Main tracking loop running in background thread."""
        while not self._stop_event.is_set():
            try:
                self._run_tracking_iteration()
                self._stop_event.wait(self.sampling_interval)
            except Exception as e:
                if self.enable_logging:
                    logging.error(f"Error in tracking loop: {e}")
                self._stop_event.wait(self.sampling_interval)

    def _trigger_alert(self, memory_mb: float, timestamp: float) -> None:
        """Trigger memory usage alert."""
        alert = {
            "timestamp": timestamp,
            "memory_mb": memory_mb,
            "threshold_mb": self.alert_threshold_mb,
            "message": f"Memory usage {memory_mb:.1f} MB exceeds threshold {self.alert_threshold_mb:.1f} MB",
        }

        with self._lock:
            self.alerts.append(alert)

        # Log alert
        if self.enable_logging:
            logging.warning(alert["message"])

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                if self.enable_logging:
                    logging.error(f"Error in alert callback: {e}")

    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback function for memory alerts."""
        self.alert_callbacks.append(callback)

    def start_tracking(self) -> None:
        """Start real-time memory tracking."""
        if self.tracking:
            if self.enable_logging:
                logging.warning("Tracking already started")
            return

        self._session_start_time = time.time()
        self._session_end_time = None
        self.tracking = True
        self._stop_event.clear()

        # Reset tracking data
        with self._lock:
            self.memory_usage.clear()
            self.timestamps.clear()
            self.events.clear()
            self.alerts.clear()
        self._last_successful_memory_mb = None
        self._set_collector_health(
            status=COLLECTOR_HEALTH_HEALTHY,
            telemetry_partial=False,
        )

        # Start tracking thread
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()

        if self.enable_logging:
            logging.info(
                f"Started memory tracking with {self.sampling_interval}s interval"
            )

    def stop_tracking(self) -> TrackingResult:
        """Stop tracking and return results."""
        if not self.tracking:
            if self.enable_logging:
                logging.warning("Tracking not started")
            return self._create_empty_result()

        self.tracking = False
        self._stop_event.set()

        # Wait for tracking thread to finish
        if self.tracking_thread:
            self.tracking_thread.join(timeout=5.0)

        self._session_end_time = time.time()
        # Create result
        result = self._create_tracking_result()

        if self.enable_logging:
            logging.info(
                f"Stopped memory tracking. Peak usage: {result.peak_memory:.1f} MB"
            )

        return result

    def _create_tracking_result(self) -> TrackingResult:
        """Create tracking result from collected data."""
        with self._lock:
            if not self.memory_usage and not self.events and not self.alerts:
                return self._create_empty_result()

            session_start = self._session_start_time
            session_end = self._session_end_time
            if session_start is None:
                session_start = self.timestamps[0] if self.timestamps else time.time()
            if session_end is None:
                session_end = self.timestamps[-1] if self.timestamps else time.time()

            return TrackingResult(
                start_time=session_start,
                end_time=session_end,
                memory_usage=self.memory_usage.copy(),
                timestamps=self.timestamps.copy(),
                events=self.events.copy(),
                alerts_triggered=self.alerts.copy(),
                peak_memory=max(self.memory_usage) if self.memory_usage else 0.0,
                average_memory=(
                    sum(self.memory_usage) / len(self.memory_usage)
                    if self.memory_usage
                    else 0.0
                ),
                min_memory=min(self.memory_usage) if self.memory_usage else 0.0,
            )

    def _create_empty_result(self) -> TrackingResult:
        """Create empty tracking result."""
        current_time = time.time()
        start_time = self._session_start_time or current_time
        end_time = self._session_end_time or start_time
        return TrackingResult(
            start_time=start_time,
            end_time=end_time,
            memory_usage=[],
            timestamps=[],
            events=[],
            alerts_triggered=[],
            peak_memory=0.0,
            average_memory=0.0,
            min_memory=0.0,
        )

    def get_current_memory(self) -> float:
        """Get current memory usage."""
        try:
            return self._get_current_memory()
        except Exception:
            return float(self._last_successful_memory_mb or 0.0)

    def get_statistics(self) -> dict[str, Any]:
        """Return current tracker health and latest successful memory sample."""
        with self._lock:
            total_events = len(self.events)
            peak_memory = max(self.memory_usage) if self.memory_usage else 0.0
            tracking_start = self._session_start_time
            tracking_end = self._session_end_time

        tracking_duration = (
            (tracking_end or time.time()) - tracking_start
            if isinstance(tracking_start, (int, float))
            else 0.0
        )
        current_memory_mb = (
            self._last_successful_memory_mb
            if self._collector_health.status == COLLECTOR_HEALTH_HEALTHY
            else None
        )
        return {
            "current_memory_mb": current_memory_mb,
            "peak_memory_mb": peak_memory,
            "total_events": total_events,
            "tracking_duration_seconds": tracking_duration,
            **self._collector_health.to_dict(),
        }

    def set_alert_threshold(self, threshold_mb: float) -> None:
        """Update alert threshold."""
        self.alert_threshold_mb = threshold_mb
        if self.enable_logging:
            logging.info(f"Updated alert threshold to {threshold_mb} MB")

    def check_alerts(self) -> bool:
        """Check if any alerts have been triggered recently."""
        with self._lock:
            # Check for alerts in the last 10 seconds
            recent_alerts = [
                alert
                for alert in self.alerts
                if time.time() - alert["timestamp"] < 10.0
            ]
            return len(recent_alerts) > 0

    def get_tracking_results(self) -> TrackingResult:
        """Get current tracking results without stopping."""
        return self._create_tracking_result()


class MemoryWatchdog:
    """Automatic memory management and cleanup for TensorFlow."""

    def __init__(
        self,
        max_memory_mb: float = 8000,
        cleanup_threshold_mb: float = 6000,
        check_interval: float = 5.0,
    ):
        """
        Initialize memory watchdog.

        Args:
            max_memory_mb: Maximum memory before forced cleanup
            cleanup_threshold_mb: Memory threshold to trigger cleanup
            check_interval: Time between memory checks in seconds
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available.")

        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold_mb = cleanup_threshold_mb
        self.check_interval = check_interval

        self.active = False
        self.watchdog_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Cleanup callbacks
        self.cleanup_callbacks: List[Callable[[], None]] = []

        logging.info(f"Memory Watchdog initialized with {max_memory_mb} MB limit")

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add cleanup callback function."""
        self.cleanup_callbacks.append(callback)

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage."""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                memory_info = tf.config.experimental.get_memory_info("/GPU:0")
                current_bytes = memory_info.get("current", 0)
                if isinstance(current_bytes, (int, float)):
                    return float(current_bytes) / (1024 * 1024)
                return 0.0
            return 0.0
        except Exception as exc:
            logging.debug("Watchdog could not get GPU memory usage: %s", exc)
            return 0.0

    def _cleanup_memory(self) -> None:
        """Perform memory cleanup."""
        try:
            # Clear TensorFlow session
            tf.keras.backend.clear_session()

            # Force garbage collection
            import gc

            gc.collect()

            # Call custom cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logging.error(f"Error in cleanup callback: {e}")

            logging.info("Performed memory cleanup")

        except Exception as e:
            logging.error(f"Error during memory cleanup: {e}")

    def _watchdog_loop(self) -> None:
        """Main watchdog loop."""
        while not self._stop_event.is_set():
            try:
                current_memory = self._get_memory_usage()

                if current_memory > self.max_memory_mb:
                    logging.warning(
                        f"Memory usage {current_memory:.1f} MB exceeds limit {self.max_memory_mb} MB - forcing cleanup"
                    )
                    self._cleanup_memory()

                elif current_memory > self.cleanup_threshold_mb:
                    logging.info(
                        f"Memory usage {current_memory:.1f} MB above threshold {self.cleanup_threshold_mb} MB - performing cleanup"
                    )
                    self._cleanup_memory()

                # Wait for next check
                self._stop_event.wait(self.check_interval)

            except Exception as e:
                logging.error(f"Error in watchdog loop: {e}")
                break

    def start(self) -> None:
        """Start memory watchdog."""
        if self.active:
            logging.warning("Watchdog already active")
            return

        self.active = True
        self._stop_event.clear()

        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()

        logging.info("Started memory watchdog")

    def stop(self) -> None:
        """Stop memory watchdog."""
        if not self.active:
            return

        self.active = False
        self._stop_event.set()

        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=5.0)

        logging.info("Stopped memory watchdog")

    def force_cleanup(self) -> None:
        """Force immediate memory cleanup."""
        self._cleanup_memory()
