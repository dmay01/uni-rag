"""Text extraction only: no OCR, image decoding, linked-resource fetches or macros."""

import re
from collections.abc import Callable
from pathlib import Path

from uni_rag.models import Unit

Warn = Callable[[str, str], None]


def markdown(text: str) -> list[Unit]:
    text = re.sub(r"\A\ufeff?---\s*\n.*?\n(?:---|\.\.\.)\s*(?:\n|$)", "", text, flags=re.S)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", lambda m: m[2] or m[1], text)
    headings: list[tuple[int, str]] = []
    units: list[Unit] = []
    body: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line) if not in_fence else None
        if match:
            if "\n".join(body).strip():
                units.append(Unit("\n".join(body).strip(), tuple(h[1] for h in headings)))
            level, title = len(match[1]), match[2]
            headings = [h for h in headings if h[0] < level] + [(level, title)]
            body = [line]
        else:
            body.append(line)
    if "\n".join(body).strip():
        units.append(Unit("\n".join(body).strip(), tuple(h[1] for h in headings)))
    return units


def parse(path: Path, warn: Warn) -> list[Unit]:
    try:
        extension = path.suffix.lower()
        if extension in {".md", ".txt", ".tex"}:
            text = path.read_text(encoding="utf-8-sig")
            return markdown(text) if extension == ".md" else [Unit(text)]
        if extension == ".pdf":
            import pymupdf

            units = []
            with pymupdf.open(path) as document:
                for number, page in enumerate(document, 1):
                    text = page.get_text().strip()
                    if text:
                        units.append(Unit(text, page=number))
                    else:
                        warn("scanned_or_empty_page", str(number))
            return units
        if extension == ".pptx":
            from pptx import Presentation

            units = []
            for number, slide in enumerate(Presentation(path).slides, 1):
                parts = [shape.text for shape in slide.shapes if shape.has_text_frame]
                for shape in slide.shapes:
                    if shape.has_table:
                        parts.extend(
                            " | ".join(cell.text for cell in row.cells) for row in shape.table.rows
                        )
                if slide.has_notes_slide:
                    notes = slide.notes_slide.notes_text_frame
                    if notes and notes.text.strip():
                        parts.append("Speaker notes: " + notes.text)
                units.append(Unit("\n".join(parts).strip(), slide=number))
            return units
        if extension == ".docx":
            from docx import Document
            from docx.table import Table
            from docx.text.paragraph import Paragraph

            units = []
            headings: list[str] = []
            body: list[str] = []
            for block in Document(path).iter_inner_content():
                if isinstance(block, Paragraph):
                    style = block.style.name if block.style else ""
                    if style.startswith("Heading ") and style[8:].isdigit():
                        if body:
                            units.append(Unit("\n".join(body), tuple(headings)))
                        level = int(style[8:])
                        headings = headings[: level - 1] + [block.text]
                        body = [block.text]
                    else:
                        body.append(block.text)
                elif isinstance(block, Table):
                    body.extend(" | ".join(c.text for c in row.cells) for row in block.rows)
            if body:
                units.append(Unit("\n".join(body), tuple(headings)))
            return units
        warn("unsupported_parser", extension)
    except Exception as exc:
        warn("parser_error", f"{type(exc).__name__}: {exc}")
    return []
