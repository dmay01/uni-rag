"""Measured CPU+GPU+ANE energy. No extrapolation, sudo, or invented missing readings."""

import bisect
import json
import logging
import math
import re
import shutil
import statistics
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from uni_rag.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    timestamp: float
    watts: float


def timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Power timestamps must include a UTC offset")
    return parsed.timestamp()


def macmon_sample(payload: dict) -> Sample:
    values = [float(payload[key]) for key in ("cpu_power", "gpu_power", "ane_power")]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Invalid power reading")
    return Sample(timestamp(payload["timestamp"]), sum(values))


def powermetrics_samples(path: Path) -> list[Sample]:
    # The external command writes text, with mW component readings and dated sample headers.
    samples = []
    current: float | None = None
    components: dict[str, float] = {}

    def flush() -> None:
        if current is not None and set(components) == {"CPU", "GPU", "ANE"}:
            values = list(components.values())
            if all(math.isfinite(v) and v >= 0 for v in values):
                samples.append(Sample(current, sum(values)))

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = re.search(r"Sampled system activity \((.*?)\)", line)
            if match:
                flush()
                current = None
                components = {}
                for fmt in ("%a %b %d %H:%M:%S.%f %Y %z", "%a %b %d %H:%M:%S %Y %z"):
                    try:
                        current = datetime.strptime(match[1], fmt).timestamp()
                        break
                    except ValueError:
                        pass
                if current is None:
                    try:
                        current = timestamp(match[1])
                    except ValueError:
                        pass
            match = re.match(r"\s*(CPU|GPU|ANE) Power:\s*([\d.eE+-]+)\s*(mW|W)\b", line)
            if match:
                components[match[1]] = float(match[2]) / (1000 if match[3] == "mW" else 1)
    flush()
    return samples


def integrate(
    samples: list[Sample], start: float, end: float, max_gap: float
) -> tuple[float | None, float | None]:
    # Coarse powermetrics timestamps may repeat: average measured powers at each timestamp.
    grouped: dict[float, list[float]] = {}
    for sample in samples:
        grouped.setdefault(sample.timestamp, []).append(sample.watts)
    ordered = [Sample(t, statistics.mean(values)) for t, values in sorted(grouped.items())]
    if (
        end <= start
        or len(ordered) < 2
        or ordered[0].timestamp > start
        or ordered[-1].timestamp < end
    ):
        return None, None
    energy = 0.0
    intervals = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        if right.timestamp <= start or left.timestamp >= end:
            continue
        gap = right.timestamp - left.timestamp
        if gap > max_gap:
            return None, None
        lo, hi = max(start, left.timestamp), min(end, right.timestamp)
        slope = (right.watts - left.watts) / gap
        p_lo = left.watts + slope * (lo - left.timestamp)
        p_hi = left.watts + slope * (hi - left.timestamp)
        energy += (p_lo + p_hi) * 0.5 * (hi - lo)
        intervals.append(gap * 1000)
    return energy, statistics.median(intervals) if intervals else None


def coverage(samples: list[Sample], start: float, end: float, max_gap: float) -> dict:
    ordered = sorted(samples, key=lambda sample: sample.timestamp)
    relevant = [sample for sample in ordered if start <= sample.timestamp <= end]
    before = [sample for sample in ordered if sample.timestamp <= start]
    after = [sample for sample in ordered if sample.timestamp >= end]
    window = ([before[-1]] if before else []) + relevant + ([after[0]] if after else [])
    times = sorted({sample.timestamp for sample in window})
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    reason = (
        "missing_start"
        if not before
        else "missing_end"
        if not after
        else "sample_gap"
        if gaps and max(gaps) > max_gap
        else None
    )
    return {
        "reason": reason,
        "max_allowed_gap_ms": max_gap * 1000,
        "sample_count": len(times),
        "first_timestamp": times[0] if times else None,
        "last_timestamp": times[-1] if times else None,
        "max_gap_ms": max(gaps) * 1000 if gaps else None,
    }


class PowerSampler:
    """One sampler per model. Preserve complete windows even if the process later exits."""

    def __init__(self, config: Config, backend: str | None = None):
        self.config = config
        self.requested = backend or config.energy_backend
        if self.requested not in {"auto", "macmon", "powermetrics", "none"}:
            raise ValueError("energy must be auto/macmon/powermetrics/none")
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.stderr_thread: threading.Thread | None = None
        self.samples: deque[Sample] = deque(maxlen=100_000)
        self.condition = threading.Condition()
        self.failed = False
        self.closing = False
        self.warnings: list[str] = []
        self.stderr: deque[str] = deque(maxlen=8)
        self.baselines: dict[str, float] = {}
        self.restarts = 0
        self.sample_log = config.data_dir / "logs" / f"macmon-{uuid.uuid4().hex[:12]}.jsonl"
        if self.requested in {"auto", "macmon"}:
            self._start_macmon()

    def warn(self, message: str) -> None:
        logger.warning(message)
        with self.condition:
            self.warnings.append(message)
        directory = self.config.data_dir / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "warnings.jsonl").open("a") as handle:
            handle.write(json.dumps({"timestamp": time.time(), "warning": message}) + "\n")

    def _start_macmon(self) -> None:
        executable = shutil.which("macmon")
        if not executable:
            self.failed = True
            self.warn("macmon unavailable; auto may use an existing current power log")
            return
        try:
            result = subprocess.run(
                [executable, "pipe", "--help"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            if "--interval" not in result.stdout:
                raise ValueError("macmon JSON pipe mode unsupported")
            self.failed = False
            self.process = subprocess.Popen(
                [executable, "pipe", "-i", str(self.config.sample_interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            process = self.process
            self.thread = threading.Thread(target=self._read, args=(process,), daemon=True)
            self.stderr_thread = threading.Thread(
                target=self._read_stderr, args=(process,), daemon=True
            )
            self.thread.start()
            self.stderr_thread.start()
            deadline = time.monotonic() + self.config.sample_wait_seconds
            with self.condition:
                while not self.samples and not self.failed:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self.condition.wait(remaining)
            if not self.samples:
                self.failed = True
                self.warn("macmon produced no usable samples before startup timeout")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.failed = True
            self.warn(f"macmon startup failed: {exc}")

    def _read_stderr(self, process: subprocess.Popen) -> None:
        for line in process.stderr:
            self.stderr.append(line.strip())

    def _read(self, process: subprocess.Popen) -> None:
        try:
            self.sample_log.parent.mkdir(parents=True, exist_ok=True)
            with self.sample_log.open("a", buffering=1) as log:
                for line in process.stdout:
                    try:
                        sample = macmon_sample(json.loads(line))
                    except (ValueError, KeyError, TypeError) as exc:
                        self.warn(f"Discarded invalid macmon record: {exc}")
                        continue
                    log.write(
                        json.dumps(
                            {
                                "timestamp": sample.timestamp,
                                "watts": sample.watts,
                                "received_at": time.time(),
                            }
                        )
                        + "\n"
                    )
                    with self.condition:
                        self.samples.append(sample)
                        self.condition.notify_all()
        except OSError as exc:
            if not self.closing:
                self.warn(f"macmon read failed: {exc}")
        finally:
            with self.condition:
                self.failed = True
                self.condition.notify_all()
            if not self.closing:
                self.warn(
                    f"macmon exited/closed stdout; return code={process.poll()}, "
                    f"stderr={list(self.stderr)}"
                )

    def prepare(self) -> None:
        """Recover between queries, never extrapolate over the failed query's gap."""
        if self.requested not in {"auto", "macmon"}:
            return
        stale = not self.samples or time.time() - self.samples[-1].timestamp > (
            self.config.sample_wait_seconds
        )
        if self.failed or stale:
            self._stop()
            with self.condition:
                self.samples.clear()
            self.restarts += 1
            self.closing = False
            self._start_macmon()

    def _window(self, start: float, end: float) -> list[Sample]:
        with self.condition:
            readings = list(self.samples)
        times = [sample.timestamp for sample in readings]
        lo = max(0, bisect.bisect_right(times, start) - 1)
        hi = bisect.bisect_left(times, end) + 1
        return readings[lo:hi]

    def measure(self, start: float, end: float) -> dict:
        result = {
            "energy_backend": "none",
            "energy_requested_backend": self.requested,
            "energy_sample_interval_ms": self.config.sample_interval_ms,
            "energy_max_allowed_gap_ms": self.config.max_sample_gap_ms,
            "energy_observed_interval_ms": None,
            "energy_gross_joules": None,
            "energy_above_baseline_joules": None,
            "energy_baseline_watts": None,
            "energy_domain": "CPU+GPU+ANE",
            "energy_warnings": [],
            "energy_restarts": self.restarts,
            "energy_sample_log": str(self.sample_log) if self.process else None,
            "energy_coverage": {},
        }
        if self.requested == "none":
            return result
        deadline = time.monotonic() + self.config.sample_wait_seconds
        warning_start = len(self.warnings)
        while True:
            choices = []
            if self.requested in {"auto", "macmon"}:
                choices.append(
                    ("macmon", self._window(start, end), self.config.max_sample_gap_ms / 1000)
                )
            if self.requested in {"auto", "powermetrics"} and self.config.power_log.exists():
                try:
                    choices.append(
                        (
                            "powermetrics",
                            powermetrics_samples(self.config.power_log),
                            max(1.25, self.config.max_sample_gap_ms / 1000),
                        )
                    )
                except (OSError, ValueError) as exc:
                    self.warn(f"Cannot read powermetrics log: {exc}")
            pending_endpoint = False
            for backend, readings, max_gap in choices:
                detail = coverage(readings, start, end, max_gap)
                result["energy_coverage"][backend] = detail
                gross, observed = integrate(readings, start, end, max_gap)
                if gross is not None:
                    baseline = self.baselines.get(backend)
                    result.update(
                        energy_backend=backend,
                        energy_max_allowed_gap_ms=max_gap * 1000,
                        energy_gross_joules=gross,
                        energy_observed_interval_ms=observed,
                        energy_baseline_watts=baseline,
                        energy_above_baseline_joules=(
                            gross - baseline * (end - start) if baseline is not None else None
                        ),
                    )
                    result["energy_warnings"] = self.warnings[warning_start:]
                    return result
                pending_endpoint |= detail["reason"] == "missing_end"
            can_wait = (self.process is not None and not self.failed) or (
                self.requested in {"auto", "powermetrics"} and self.config.power_log.exists()
            )
            if not pending_endpoint or not can_wait or time.monotonic() >= deadline:
                break
            with self.condition:
                self.condition.wait(min(0.1, max(0, deadline - time.monotonic())))
        detail = {
            "query_start": start,
            "query_end": end,
            "coverage": result["energy_coverage"],
            "stderr": list(self.stderr),
            "process_exit": self.process.poll() if self.process else None,
        }
        self.warn("Incomplete power coverage; energy is null: " + json.dumps(detail))
        result["energy_warnings"] = self.warnings[warning_start:]
        return result

    def measure_idle(self, seconds: float) -> dict:
        if seconds <= 0 or self.requested == "none":
            return {}
        self.prepare()
        start = time.time()
        time.sleep(seconds)
        end = time.time()
        measurement = self.measure(start, end)
        if measurement["energy_gross_joules"] is not None:
            self.baselines[measurement["energy_backend"]] = measurement["energy_gross_joules"] / (
                end - start
            )
        return measurement

    def _stop(self) -> None:
        self.closing = True
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            for thread in (self.thread, self.stderr_thread):
                if thread:
                    thread.join(timeout=1)
            for pipe in (self.process.stdout, self.process.stderr):
                if pipe:
                    pipe.close()
            self.process = None

    def close(self) -> None:
        self._stop()

    def __enter__(self) -> "PowerSampler":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
