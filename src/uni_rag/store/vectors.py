"""Explicit LanceDB schema; embeddings are supplied locally, never auto-generated."""

import json
from pathlib import Path

import lancedb
import pyarrow as pa

from uni_rag.models import Chunk


class VectorStore:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(directory))
        self.table = self.db.open_table("chunks") if "chunks" in self.db.table_names() else None

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts differ")
        if not chunks:
            return
        rows = [
            {"id": c.id, "text": c.text, "metadata": json.dumps(c.metadata()), "vector": v}
            for c, v in zip(chunks, vectors, strict=True)
        ]
        if self.table is None:
            schema = pa.schema(
                [
                    ("id", pa.string()),
                    ("text", pa.string()),
                    ("metadata", pa.string()),
                    ("vector", pa.list_(pa.float32(), len(vectors[0]))),
                ]
            )
            self.table = self.db.create_table("chunks", data=rows, schema=schema)
        else:
            self.table.merge_insert(
                "id"
            ).when_matched_update_all().when_not_matched_insert_all().execute(rows)

    def chunks(self) -> list[Chunk]:
        if self.table is None:
            return []
        return [
            Chunk.from_row(row)
            for row in self.table.search().select(["id", "text", "metadata"]).limit(None).to_list()
        ]

    def search(self, vector: list[float], k: int) -> list[Chunk]:
        if self.table is None:
            return []
        rows = self.table.search(vector).metric("cosine").limit(k).to_list()
        return [Chunk.from_row(row) for row in rows]

    def clear(self) -> None:
        if self.table is not None:
            self.db.drop_table("chunks")
            self.table = None
