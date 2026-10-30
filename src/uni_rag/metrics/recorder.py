"""Append-only, lock-protected JSONL, without question or document text."""

import fcntl
import json
from datetime import UTC, datetime

from uni_rag.config import Config


def record(config: Config, metric: dict) -> None:
    directory = config.data_dir / "metrics"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"queries-{datetime.now(UTC):%Y-%m-%d}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(metric, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)
