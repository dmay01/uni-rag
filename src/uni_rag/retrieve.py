"""Filtered dense and BM25 retrieval, fused by reciprocal rank (RRF)."""

import hashlib
import json
from dataclasses import replace

from uni_rag.config import Config
from uni_rag.ingest.manifest import active_files, validate_index
from uni_rag.models import Chunk, Hit
from uni_rag.ollama import Ollama
from uni_rag.store.keywords import KeywordStore
from uni_rag.store.vectors import VectorStore


def active_chunks(config: Config, store: VectorStore) -> tuple[list[Chunk], str]:
    configured = {source.name for source in config.sources}
    files = [item for item in active_files(config.data_dir) if item["source"] in configured]
    by_hash: dict[str, list[dict]] = {}
    for item in files:
        by_hash.setdefault(item["sha1"], []).append(item)
    chunks = []
    for chunk in store.chunks():
        if aliases := by_hash.get(chunk.content_hash):
            template = chunk.locations[0]
            chunk.locations = [
                replace(
                    template,
                    source=f["source"],
                    path=f["path"],
                    file_type="." + f["path"].rsplit(".", 1)[-1].lower(),
                    mtime_ns=f["mtime_ns"],
                )
                for f in aliases
            ]
            chunks.append(chunk)
    revision = hashlib.sha1(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return chunks, revision


class Retriever:
    """Load the corpus and keyword index once for a batch of independent questions."""

    def __init__(self, config: Config, client: Ollama):
        validate_index(config)
        self.config, self.client = config, client
        self.store = VectorStore(config.data_dir / "index")
        self.chunks, revision = active_chunks(config, self.store)
        self.keyword = KeywordStore(config.data_dir / "bm25", self.chunks, revision)

    def search(
        self,
        question: str,
        k: int | None = None,
        source: str | None = None,
        file_type: str | None = None,
        path_prefix: str | None = None,
    ) -> list[Hit]:
        config = self.config
        count = config.top_k if k is None else k
        if count < 1:
            raise ValueError("k must be positive")
        if source:
            config.selected_sources(source)
        if file_type:
            file_type = "." + file_type.lstrip(".").lower()
        eligible = {}
        for chunk in self.chunks:
            locations = [
                loc
                for loc in chunk.locations
                if (source is None or loc.source == source)
                and (file_type is None or loc.file_type == file_type)
                and (path_prefix is None or loc.path.startswith(path_prefix))
            ]
            if locations:
                eligible[chunk.id] = replace(chunk, locations=locations)
        if not eligible:
            return []
        candidates = max(count, config.candidate_k)
        vector = self.client.embed([question])[0]
        vector_filter = None if len(eligible) == self.store.count() else set(eligible)
        dense = [chunk.id for chunk in self.store.search(vector, candidates, vector_filter)]
        sparse = self.keyword.search(question, candidates, set(eligible))
        scores: dict[str, float] = {}
        for ranking in (dense, sparse):
            for rank, identifier in enumerate(ranking, 1):
                scores[identifier] = scores.get(identifier, 0) + 1 / (config.rrf_constant + rank)
        ordered = sorted(scores, key=lambda identifier: (-scores[identifier], identifier))[:count]
        return [Hit(eligible[identifier], scores[identifier]) for identifier in ordered]


def retrieve(
    question: str,
    config: Config,
    client: Ollama,
    k: int | None = None,
    source: str | None = None,
    file_type: str | None = None,
    path_prefix: str | None = None,
) -> list[Hit]:
    return Retriever(config, client).search(question, k, source, file_type, path_prefix)
