"""Readable ingestion orchestration with persistent skip logs and batched writes."""

import hashlib
import json
import time
from pathlib import Path

from rich.progress import Progress, TimeRemainingColumn

from uni_rag.config import Config
from uni_rag.ingest.chunker import chunk
from uni_rag.ingest.parsers import parse
from uni_rag.ingest.walker import walk
from uni_rag.ollama import Ollama
from uni_rag.store.vectors import VectorStore


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def ingest(
    config: Config,
    source: str | None = None,
    only: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    ollama: Ollama | None = None,
) -> dict:
    start = time.perf_counter()
    scan = walk(config, source, only, limit)
    summary = scan.summary()
    summary.update({"dry_run": dry_run, "files_indexed": 0, "chunks": 0})
    if not dry_run:
        store = VectorStore(config.data_dir / "index")
        with Ollama(config) as local:
            client = ollama or local
            with Progress(*Progress.get_default_columns(), TimeRemainingColumn()) as progress:
                task = progress.add_task("Indexing files", total=len(scan.files))
                for item in scan.files:
                    try:
                        digest = hashlib.sha1(item.path.read_bytes()).hexdigest()
                        units = parse(
                            item.path,
                            lambda reason, detail, item=item: scan.skip(
                                item.source, item.relative_path, reason, detail
                            ),
                        )
                        chunks = chunk(units, item, digest, config)
                        for offset in range(0, len(chunks), config.embedding_batch_size):
                            batch = chunks[offset : offset + config.embedding_batch_size]
                            store.upsert(batch, client.embed([c.text for c in batch]))
                        summary["files_indexed"] += bool(chunks)
                        summary["chunks"] += len(chunks)
                    except OSError as exc:
                        scan.skip(item.source, item.relative_path, "read_error", str(exc))
                    progress.advance(task)
    summary["skipped_by_reason"] = scan.skip_counts
    summary["duration_seconds"] = time.perf_counter() - start
    log_dir = config.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "skipped.jsonl").open("a", encoding="utf-8") as handle:
        for row in scan.skipped:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(config.data_dir / "reports/ingest.json", summary)
    return summary
