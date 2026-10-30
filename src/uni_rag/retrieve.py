"""Retrieval with explicit context records."""

from uni_rag.config import Config
from uni_rag.models import Hit
from uni_rag.ollama import Ollama
from uni_rag.store.vectors import VectorStore


def retrieve(
    question: str,
    config: Config,
    client: Ollama,
    k: int | None = None,
    source: str | None = None,
    file_type: str | None = None,
    path_prefix: str | None = None,
) -> list[Hit]:
    count = k or config.top_k
    if count < 1:
        raise ValueError("k must be positive")
    store = VectorStore(config.data_dir / "index")
    chunks = store.search(client.embed([question])[0], max(count, config.candidate_k))
    results = []
    for chunk in chunks:
        chunk.locations = [
            loc
            for loc in chunk.locations
            if (source is None or loc.source == source)
            and (file_type is None or loc.file_type == file_type)
            and (path_prefix is None or loc.path.startswith(path_prefix))
        ]
        if chunk.locations:
            results.append(Hit(chunk, 1 / (len(results) + 1)))
    return results[:count]
