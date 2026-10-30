"""Structure-first windows using a documented, conservative token estimate."""

import hashlib
import re

from uni_rag.config import Config
from uni_rag.models import Chunk, FileRecord, Location, Unit


def windows(text: str, target: int, overlap: float) -> list[str]:
    # Approximately four Unicode characters per token; whitespace boundaries are preferred.
    max_chars = target * 4
    result = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundaries = list(re.finditer(r"\s+", text[start:end]))
            if boundaries and boundaries[-1].start() > max_chars // 2:
                end = start + boundaries[-1].start()
        part = text[start:end].strip()
        if part:
            result.append(part)
        if end == len(text):
            break
        start = max(start + 1, end - int(max_chars * overlap))
    return result


def chunk(units: list[Unit], item: FileRecord, content_hash: str, config: Config) -> list[Chunk]:
    grouped: list[tuple[Unit, int | None]] = []
    index = 0
    while index < len(units):
        unit = units[index]
        slide_end = unit.slide
        # Adjacent short slides only; retain the full inclusive slide range for citation.
        if unit.slide is not None and len(unit.text) < config.min_slide_tokens * 4:
            if index + 1 < len(units):
                following = units[index + 1]
                unit = Unit(unit.text + "\n\n" + following.text, slide=unit.slide)
                slide_end = following.slide
                index += 1
        grouped.append((unit, slide_end))
        index += 1
    result = []
    for unit, slide_end in grouped:
        texts = (
            [unit.text]
            if unit.slide
            else windows(unit.text, config.target_tokens, config.overlap_fraction)
        )
        for text in texts:
            if not text.strip():
                continue
            location = Location(
                item.source,
                item.relative_path,
                item.path.suffix.lower(),
                unit.page,
                unit.slide,
                slide_end,
                unit.heading_path,
                item.mtime_ns,
            )
            identifier = hashlib.sha1(f"{content_hash}:{len(result)}".encode()).hexdigest()
            result.append(Chunk(identifier, text.strip(), content_hash, [location]))
    return result
