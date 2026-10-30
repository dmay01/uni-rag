"""SQLite is the authority for completed files; LanceDB batches can be safely retried."""

import hashlib
import json
import sqlite3
from pathlib import Path

from uni_rag.config import Config
from uni_rag.models import FileRecord


class Manifest:
    def __init__(self, directory: Path, config: Config):
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "manifest.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS files (
                source TEXT, path TEXT, size INTEGER, mtime_ns INTEGER, sha1 TEXT,
                PRIMARY KEY (source, path));
            CREATE TABLE IF NOT EXISTS contents (sha1 TEXT PRIMARY KEY, chunks INTEGER);
        """)
        signature = json.dumps(
            {
                "embedding_model": config.embedding_model,
                "target_tokens": config.target_tokens,
                "overlap": config.overlap_fraction,
                "min_slide_tokens": config.min_slide_tokens,
                "parser_version": 1,
            },
            sort_keys=True,
        )
        old = self.db.execute("SELECT value FROM settings WHERE key='signature'").fetchone()
        if old and old[0] != signature:
            self.db.close()
            raise ValueError("Embedding/chunk settings changed. Use a fresh storage.data_dir.")
        self.db.execute("INSERT OR IGNORE INTO settings VALUES ('signature', ?)", (signature,))
        self.db.commit()

    def __enter__(self) -> "Manifest":
        return self

    def __exit__(self, *args: object) -> None:
        self.db.close()

    def get(self, source: str, path: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM files WHERE source=? AND path=?", (source, path)
        ).fetchone()
        return dict(row) if row else None

    def all_files(self) -> list[dict]:
        return [dict(row) for row in self.db.execute("SELECT * FROM files ORDER BY source, path")]

    def complete(self, digest: str) -> bool:
        return (
            self.db.execute("SELECT 1 FROM contents WHERE sha1=?", (digest,)).fetchone() is not None
        )

    def chunk_count(self, digest: str) -> int:
        row = self.db.execute("SELECT chunks FROM contents WHERE sha1=?", (digest,)).fetchone()
        return int(row[0]) if row else 0

    def mark(self, item: FileRecord, digest: str, chunks: int) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO contents VALUES (?, ?)", (digest, chunks))
            self.db.execute(
                "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?)",
                (item.source, item.relative_path, item.size, item.mtime_ns, digest),
            )

    def remove(self, source: str, path: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM files WHERE source=? AND path=?", (source, path))

    def forget_unreferenced(self) -> None:
        with self.db:
            self.db.execute("DELETE FROM contents WHERE sha1 NOT IN (SELECT sha1 FROM files)")

    def assert_roots(self, config: Config) -> None:
        for source in config.sources:
            key = f"root:{source.name}"
            path = str(source.path.resolve())
            old = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            if old and old[0] != path:
                raise ValueError(f"Root for {source.name} changed. Use a fresh storage.data_dir.")
            self.db.execute("INSERT OR IGNORE INTO settings VALUES (?, ?)", (key, path))
        self.db.commit()

    @property
    def revision(self) -> str:
        return hashlib.sha1(json.dumps(self.all_files(), sort_keys=True).encode()).hexdigest()


def active_files(directory: Path) -> list[dict]:
    path = directory / "manifest.sqlite3"
    if not path.exists():
        return []
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT * FROM files ORDER BY source, path")]


def validate_index(config: Config) -> None:
    path = config.data_dir / "manifest.sqlite3"
    if not path.exists():
        return
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        settings = dict(db.execute("SELECT key, value FROM settings"))
    signature = json.loads(settings.get("signature", "{}"))
    if signature.get("embedding_model") != config.embedding_model:
        raise ValueError("Embedding model differs from the index. Use a fresh storage.data_dir.")
    for source in config.sources:
        root = settings.get(f"root:{source.name}")
        if root is not None and root != str(source.path.resolve()):
            raise ValueError(
                f"Root for {source.name} differs from the index. Use a fresh data_dir."
            )
