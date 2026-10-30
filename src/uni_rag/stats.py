"""Local index and query statistics; never enumerates source directories."""

import json
from collections import Counter

from uni_rag.config import Config
from uni_rag.ingest.manifest import active_files
from uni_rag.retrieve import active_chunks
from uni_rag.store.vectors import VectorStore


def stats(config: Config) -> dict:
    files = active_files(config.data_dir)
    chunks, _ = active_chunks(config, VectorStore(config.data_dir / "index"))
    hashes = Counter(item["sha1"] for item in files)
    statuses = Counter()
    malformed = 0
    for path in (config.data_dir / "metrics").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                statuses[json.loads(line).get("status", "unknown")] += 1
            except json.JSONDecodeError:
                malformed += 1
    return {
        "files": len(files),
        "chunks": len(chunks),
        "unique_contents": len(hashes),
        "duplicate_files": sum(count - 1 for count in hashes.values()),
        "files_by_source": dict(Counter(item["source"] for item in files)),
        "indexed_bytes_including_duplicates": sum(item["size"] for item in files),
        "queries_by_status": dict(statuses),
        "malformed_metric_lines": malformed,
        "data_dir": str(config.data_dir),
    }
