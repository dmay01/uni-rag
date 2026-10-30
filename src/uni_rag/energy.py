"""Measured CPU+GPU+ANE energy. No extrapolation, sudo, or invented missing readings."""

import json
import logging
import math
import re
import shutil
import statistics
import subprocess
import threading
import time
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


class PowerSampler:
    """One live sampler per benchmark; ask/chat never measure an idle baseline."""

    def __init__(self, config: Config, backend: str | None = None):
        self.config = config
        self.requested = backend or config.energy_backend
        if self.requested not in {"auto", "macmon", "powermetrics", "none"}:
            raise ValueError("energy must be auto/macmon/powermetrics/none")
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.samples: deque[Sample] = deque(maxlen=100_000)
        self.failed = False
        self.warnings: list[str] = []
        self.baselines: dict[str, float] = {}
        if self.requested in {"auto", "macmon"}:
            self._start_macmon()

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            logger.warning(message)
            self.warnings.append(message)
            directory = self.config.data_dir / "logs"
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "warnings.jsonl").open("a") as handle:
                handle.write(json.dumps({"timestamp": time.time(), "warning": message}) + "\n")

    def _start_macmon(self) -> None:
        executable = shutil.which("macmon")
        if not executable:
            self.warn("macmon is unavailable; trying an existing power log if configured")
            return
        try:
            help_result = subprocess.run(
                [executable, "pipe", "--help"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            if "--interval" not in help_result.stdout:
                raise ValueError("macmon pipe JSON mode is unsupported")
            self.process = subprocess.Popen(
                [executable, "pipe", "-i", str(self.config.sample_interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
            deadline = time.monotonic() + max(1.5, self.config.sample_interval_ms / 500)
            while not self.samples and not self.failed and time.monotonic() < deadline:
                time.sleep(0.01)
            if not self.samples:
                self.failed = True
                self.warn("macmon produced no usable samples; energy may be unavailable")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.failed = True
            self.warn(f"macmon failed: {exc}; continuing without live power readings")

    def _read(self) -> None:
        try:
            for line in self.process.stdout:
                self.samples.append(macmon_sample(json.loads(line)))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.failed = True
            self.warn(f"macmon sampling failed: {exc}; query will continue")
        finally:
            if self.process and self.process.poll() is not None:
                self.failed = True

    def measure(self, start: float, end: float) -> dict:
        result = {
            "energy_backend": "none",
            "energy_requested_backend": self.requested,
            "energy_sample_interval_ms": self.config.sample_interval_ms,
            "energy_observed_interval_ms": None,
            "energy_gross_joules": None,
            "energy_above_baseline_joules": None,
            "energy_baseline_watts": None,
            "energy_domain": "CPU+GPU+ANE",
            "energy_warnings": [],
        }
        if self.requested == "none":
            return result
        deadline = time.monotonic() + max(1.3, 3 * self.config.sample_interval_ms / 1000)
        while True:
            choices = []
            if self.requested in {"auto", "macmon"} and self.samples and not self.failed:
                choices.append(
                    ("macmon", list(self.samples), 1.75 * self.config.sample_interval_ms / 1000)
                )
            if self.requested in {"auto", "powermetrics"} and self.config.power_log.exists():
                try:
                    # Read repeatedly until the external writer has flushed the trailing sample.
                    choices.append(
                        (
                            "powermetrics",
                            powermetrics_samples(self.config.power_log),
                            max(1.25, 1.75 * self.config.sample_interval_ms / 1000),
                        )
                    )
                except (OSError, ValueError) as exc:
                    self.warn(f"Cannot read powermetrics log: {exc}")
            for backend, readings, max_gap in choices:
                gross, observed = integrate(readings, start, end, max_gap)
                if gross is not None:
                    baseline = self.baselines.get(backend)
                    result.update(
                        energy_backend=backend,
                        energy_gross_joules=gross,
                        energy_observed_interval_ms=observed,
                        energy_baseline_watts=baseline,
                        energy_above_baseline_joules=(
                            gross - baseline * (end - start) if baseline is not None else None
                        ),
                    )
                    result["energy_warnings"] = list(self.warnings)
                    return result
            if time.monotonic() >= deadline or (
                not choices and not self.process and not self.config.power_log.exists()
            ):
                break
            time.sleep(0.02)
        self.warn("No complete, current power coverage for the query window; energy is null")
        result["energy_warnings"] = list(self.warnings)
        return result

    def measure_idle(self, seconds: float) -> dict:
        """Explicitly invoked by eval only, after model warm-up."""
        if seconds <= 0 or self.requested == "none":
            return {}
        start = time.time()
        time.sleep(seconds)
        end = time.time()
        measurement = self.measure(start, end)
        if measurement["energy_gross_joules"] is not None:
            self.baselines[measurement["energy_backend"]] = measurement["energy_gross_joules"] / (
                end - start
            )
        return measurement

    def close(self) -> None:
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            if self.thread:
                self.thread.join(timeout=1)
            if self.process.stdout:
                self.process.stdout.close()

    def __enter__(self) -> "PowerSampler":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
