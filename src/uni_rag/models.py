"""Small shared data structures, serialized without framework-specific objects."""

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FileRecord:
    source: str
    path: Path
    relative_path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class Unit:
    text: str
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    slide: int | None = None


@dataclass(frozen=True)
class Location:
    source: str
    path: str
    file_type: str
    page: int | None = None
    slide: int | None = None
    slide_end: int | None = None
    heading_path: tuple[str, ...] = ()
    mtime_ns: int = 0

    def label(self) -> str:
        suffix = f", page {self.page}" if self.page else ""
        if self.slide:
            suffix = f", slide {self.slide}"
            if self.slide_end and self.slide_end != self.slide:
                suffix += f"–{self.slide_end}"
        return f"{self.source}: {self.path}{suffix}"


@dataclass
class Chunk:
    id: str
    text: str
    content_hash: str
    locations: list[Location] = field(default_factory=list)

    def metadata(self) -> dict:
        return {"content_hash": self.content_hash, "locations": [asdict(x) for x in self.locations]}

    @classmethod
    def from_row(cls, row: dict) -> "Chunk":
        import json

        meta = json.loads(row["metadata"])
        return cls(
            row["id"],
            row["text"],
            meta["content_hash"],
            [Location(**loc) for loc in meta["locations"]],
        )


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    metrics: dict
