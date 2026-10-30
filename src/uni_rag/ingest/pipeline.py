"""Incremental file commits; idempotent embedding batches survive interruption."""

import hashlib
import json
import time
from pathlib import Path, PurePosixPath

from rich.progress import Progress, TimeRemainingColumn

from uni_rag.config import Config
from uni_rag.ingest.chunker import chunk
from uni_rag.ingest.manifest import Manifest
from uni_rag.ingest.parsers import parse
from uni_rag.ingest.walker import PERMISSION_HINT, SF_DATALESS, Scan, walk
from uni_rag.ollama import Ollama
from uni_rag.store.keywords import KeywordStore
from uni_rag.store.vectors import VectorStore


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prune(
    manifest: Manifest,
    scan: Scan,
    config: Config,
    source: str | None,
    only: str | None,
    limit: int | None,
) -> int:
    if limit is not None:
        return 0  # A limited scan cannot establish that any file was deleted.
    selected = {s.name for s in config.selected_sources(source)}
    eligible = {(f.source, f.relative_path) for f in scan.files}
    protected_reasons = {"permission", "unavailable", "cloud_only", "stat_error"}
    removed = 0
    for old in manifest.all_files():
        if old["source"] not in selected:
            continue
        if only and not PurePosixPath(old["path"]).is_relative_to(PurePosixPath(only)):
            continue
        if (old["source"], old["path"]) in eligible:
            continue
        if any(
            s["source"] == old["source"]
            and s["reason"] in protected_reasons
            and (s["path"] == "." or PurePosixPath(old["path"]).is_relative_to(s["path"]))
            for s in scan.skipped
        ):
            continue
        manifest.remove(old["source"], old["path"])
        removed += 1
    return removed


def save_report(config: Config, scan: Scan, summary: dict, start: float) -> dict:
    summary["skipped_by_reason"] = scan.skip_counts
    summary["duration_seconds"] = time.perf_counter() - start
    log_dir = config.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "skipped.jsonl").open("a", encoding="utf-8") as handle:
        for row in scan.skipped:
            handle.write(json.dumps({"timestamp": time.time(), **row}, ensure_ascii=False) + "\n")
    write_json(config.data_dir / "reports/ingest.json", summary)
    return summary


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
    summary = {
        **scan.summary(),
        "dry_run": dry_run,
        "files_indexed": 0,
        "chunks": 0,
        "files_unchanged": 0,
        "duplicate_files": 0,
        "files_removed": 0,
        "embedding_chunks_written": 0,
        "status": "complete",
    }
    if dry_run:
        return save_report(config, scan, summary, start)
    try:
        with Manifest(config.data_dir, config) as manifest, Ollama(config) as local:
            manifest.assert_roots(config)
            store = VectorStore(config.data_dir / "index")
            existing = store.ids()
            client = ollama or local
            with Progress(*Progress.get_default_columns(), TimeRemainingColumn()) as progress:
                task = progress.add_task("Indexing files", total=len(scan.files))
                for item in scan.files:
                    try:
                        old = manifest.get(item.source, item.relative_path)
                        if old and (old["size"], old["mtime_ns"]) == (item.size, item.mtime_ns):
                            summary["files_unchanged"] += 1
                            continue
                        info = item.path.stat(follow_symlinks=False)
                        if item.path.is_symlink():
                            scan.skip(item.source, item.relative_path, "symlink")
                            continue
                        if config.skip_cloud_only and getattr(info, "st_flags", 0) & SF_DATALESS:
                            scan.skip(item.source, item.relative_path, "cloud_only")
                            continue
                        if (info.st_size, info.st_mtime_ns) != (item.size, item.mtime_ns):
                            scan.skip(item.source, item.relative_path, "changed_during_ingest")
                            continue
                        # Hash streaming avoids loading an entire large file just for deduplication.
                        with item.path.open("rb") as handle:
                            digest = hashlib.file_digest(handle, "sha1").hexdigest()
                        if manifest.complete(digest):
                            final = item.path.stat(follow_symlinks=False)
                            if (final.st_size, final.st_mtime_ns) != (item.size, item.mtime_ns):
                                scan.skip(item.source, item.relative_path, "changed_during_ingest")
                                continue
                            count = manifest.chunk_count(digest)
                            manifest.mark(item, digest, count)
                            summary["duplicate_files"] += 1
                            continue
                        warnings: list[tuple[str, str]] = []
                        units = parse(
                            item.path,
                            lambda reason, detail, warnings=warnings: warnings.append(
                                (reason, detail)
                            ),
                        )
                        for reason, detail in warnings:
                            scan.skip(item.source, item.relative_path, reason, detail)
                        if any(reason == "parser_error" for reason, _ in warnings):
                            continue
                        chunks = chunk(units, item, digest, config)
                        pending = [c for c in chunks if c.id not in existing]
                        embedding_task = progress.add_task(
                            "Embedding chunks", total=len(pending), visible=bool(pending)
                        )
                        try:
                            for offset in range(0, len(pending), config.embedding_batch_size):
                                batch = pending[offset : offset + config.embedding_batch_size]
                                store.upsert(batch, client.embed([c.text for c in batch]))
                                existing.update(c.id for c in batch)
                                summary["embedding_chunks_written"] += len(batch)
                                progress.advance(embedding_task, len(batch))
                        finally:
                            progress.remove_task(embedding_task)
                        final = item.path.stat(follow_symlinks=False)
                        if (final.st_size, final.st_mtime_ns) != (item.size, item.mtime_ns):
                            scan.skip(item.source, item.relative_path, "changed_during_ingest")
                            continue
                        manifest.mark(item, digest, len(chunks))
                        summary["files_indexed"] += bool(chunks)
                        summary["chunks"] += len(chunks)
                    except OSError as exc:
                        scan.skip(
                            item.source,
                            item.relative_path,
                            "read_error",
                            f"{exc}. {PERMISSION_HINT}"
                            if isinstance(exc, PermissionError)
                            else str(exc),
                        )
                    finally:
                        progress.advance(task)
            summary["files_removed"] = prune(manifest, scan, config, source, only, limit)
            files = manifest.all_files()
            active = store.reconcile(files)
            active_hashes = {f["sha1"] for f in files}
            # Collect orphans only after successful completion; preserve interrupted batches.
            store.delete({c.id for c in store.chunks() if c.content_hash not in active_hashes})
            manifest.forget_unreferenced()
            KeywordStore(config.data_dir / "bm25", active, manifest.revision)
            summary["total_chunks"] = len(active)
    except BaseException:
        summary["status"] = "interrupted_or_failed"
        raise
    finally:
        save_report(config, scan, summary, start)
    return summary
