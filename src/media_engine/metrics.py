"""Bounded process-lifetime Prometheus counters, without request or model labels."""

import math
import threading
import time
from collections import defaultdict

PHASES = frozenset(("encoding", "sampling", "decoding", "packaging", "saving", "upscaling"))


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.executions = defaultdict(int)
        self.phases = defaultdict(float)
        self.load_seconds = 0.0
        self.busy_refusals = 0
        self.peak_memory = None
        self.phase = None  # single-flight worker: at most one active phase

    def stage(self, task, phase):
        with self.lock:
            self._finish_phase()
            self.phase = (task, phase if phase in PHASES else "other", time.monotonic())

    def _finish_phase(self):
        if self.phase is not None:
            task, phase, started = self.phase
            self.phases[task, phase] += max(0.0, time.monotonic() - started)
            self.phase = None

    def finish(self, task, state, measurements):
        with self.lock:
            self._finish_phase()
            self.executions[task, state] += 1
            value = (measurements or {}).get("peak_allocated_mb")
            if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                self.peak_memory = max(self.peak_memory or 0, value * 1024 * 1024)

    def load(self, elapsed):
        with self.lock:
            self.load_seconds += elapsed

    def refused(self):
        with self.lock:
            self.busy_refusals += 1

    def render(self, busy):
        with self.lock:
            lines = [
                "# TYPE media_engine_executions_total counter",
                "# TYPE media_engine_phase_seconds_total counter",
                "# TYPE media_engine_load_seconds_total counter",
                "# TYPE media_engine_busy_refusals_total counter",
                "# TYPE media_engine_busy gauge",
                f"media_engine_load_seconds_total {self.load_seconds:.9f}",
                f"media_engine_busy_refusals_total {self.busy_refusals}",
                f"media_engine_busy {int(busy)}",
            ]
            for (task, state), value in sorted(self.executions.items()):
                lines.append(f'media_engine_executions_total{{task="{task}",state="{state}"}} {value}')
            for (task, phase), value in sorted(self.phases.items()):
                lines.append(f'media_engine_phase_seconds_total{{task="{task}",phase="{phase}"}} {value:.9f}')
            # Missing device measurements are omitted, not misrepresented as zero.
            if self.peak_memory is not None:
                lines.extend(("# TYPE media_engine_peak_memory_bytes gauge", f"media_engine_peak_memory_bytes {self.peak_memory:.0f}"))
            return "\n".join(lines) + "\n"
