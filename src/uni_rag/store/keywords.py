"""BM25s with a local, revisioned cache. No language models or downloads."""

import json
import shutil
from pathlib import Path

import bm25s
import numpy as np

from uni_rag.models import Chunk


class KeywordStore:
    def __init__(self, directory: Path, chunks: list[Chunk], revision: str):
        self.chunks = sorted(chunks, key=lambda item: item.id)
        self.index = None
        if not chunks:
            return
        marker = directory / "revision.json"
        expected = {"revision": revision, "ids": [c.id for c in self.chunks]}
        if marker.exists():
            try:
                if json.loads(marker.read_text()) == expected:
                    self.index = bm25s.BM25.load(str(directory), mmap=True, verbosity=0)
                    return
            except (OSError, ValueError, KeyError):
                pass  # A crash while saving the disposable cache is repaired locally.
        tokens = bm25s.tokenize([c.text for c in self.chunks], stopwords=[], show_progress=False)
        self.index = bm25s.BM25()
        self.index.index(tokens, show_progress=False)
        staging = directory.with_name(directory.name + ".building")
        if staging.exists():
            shutil.rmtree(staging)
        self.index.save(str(staging), show_progress=False)
        (staging / "revision.json").write_text(json.dumps(expected))
        if directory.exists():
            shutil.rmtree(directory)
        staging.rename(directory)

    def search(self, question: str, k: int, eligible: set[str]) -> list[str]:
        if self.index is None or not eligible:
            return []
        tokens = bm25s.tokenize(question, stopwords=[], show_progress=False)
        # A zero mask removes filtered documents before ranking, including duplicate aliases.
        mask = np.array([float(c.id in eligible) for c in self.chunks], dtype=np.float32)
        ids, scores = self.index.retrieve(
            tokens, k=min(k, len(self.chunks)), weight_mask=mask, show_progress=False
        )
        return [
            self.chunks[int(i)].id
            for i, score in zip(ids[0], scores[0], strict=True)
            if score > 0 and self.chunks[int(i)].id in eligible
        ]
