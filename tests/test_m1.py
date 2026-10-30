import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from uni_rag.config import Config, Source
from uni_rag.generate import ask
from uni_rag.ingest.chunker import chunk, windows
from uni_rag.ingest.parsers import parse
from uni_rag.ingest.pipeline import ingest
from uni_rag.ingest.walker import SF_DATALESS, walk
from uni_rag.models import FileRecord
from uni_rag.ollama import Ollama


def transport(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    if request.url.path == "/api/embed":
        return httpx.Response(200, json={"embeddings": [[1.0, 0.5, 0.1] for _ in payload["input"]]})
    assert payload["think"] is False
    assert "Hybrid retrieval" in payload["messages"][1]["content"]
    return httpx.Response(
        200,
        text='{"message":{"content":"Vectors and BM25 [1]."}}\n'
        '{"done":true,"prompt_eval_count":80,"eval_count":8}\n',
    )


def test_fixture_ingest_and_ask(cfg):
    with Ollama(cfg, httpx.MockTransport(transport)) as client:
        report = ingest(cfg, ollama=client)
        assert report["files_indexed"] == 6
        assert report["skipped_by_reason"]["scanned_or_empty_page"] == 1
        answer = ask("What is hybrid retrieval?", cfg, client)
    assert "[1]" in answer.text
    assert answer.metrics["prompt_tokens"] == 80
    assert answer.hits


def test_dry_run_never_reads_content(cfg, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Document content was read")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    report = ingest(cfg, dry_run=True)
    assert report["eligible_files"] == 6
    assert not (cfg.data_dir / "index").exists()


def test_walker_exclusions(cfg):
    root = cfg.sources[0].path
    (root / ".obsidian").mkdir()
    (root / ".obsidian/secret.md").write_text("ignored")
    (root / "pending.md.icloud").touch()
    (root / "image.png").write_bytes(b"not an image")
    (root / "link.md").symlink_to(root / "notes.md")
    scan = walk(cfg)
    assert scan.skip_counts == {
        "excluded_directory": 1,
        "extension": 1,
        "symlink": 1,
        "icloud_placeholder": 1,
    }
    assert len(scan.files) == 6
    with pytest.raises(ValueError):
        walk(cfg, only="../escape")


def test_dataless_does_not_open_directory(cfg, monkeypatch):
    import uni_rag.ingest.walker as walker

    class Entry:
        name = "cloud.md"
        path = str(cfg.sources[0].path / name)

        def is_symlink(self):
            return False

        def stat(self, **kwargs):
            return SimpleNamespace(st_mode=0o100644, st_flags=SF_DATALESS, st_size=1, st_mtime_ns=0)

    class Entries:
        def __enter__(self):
            return iter([Entry()])

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(walker.os, "scandir", lambda path: Entries())
    result = walk(cfg)
    assert not result.files
    assert result.skip_counts == {"cloud_only": 1}


def test_parsers_and_metadata(cfg):
    warnings = []
    for item in walk(cfg).files:
        units = parse(item.path, lambda *args: warnings.append(args))
        chunks = chunk(units, item, "digest", cfg)
        assert chunks
        assert all(c.locations[0].mtime_ns for c in chunks)
        if item.path.suffix == ".md":
            text = " ".join(c.text for c in chunks)
            assert "private_tag" not in text and "[[" not in text and "diagram.png" not in text
            assert chunks[-1].locations[0].heading_path == ("Retrieval", "Evaluation")
        if item.path.suffix == ".pptx":
            assert "bge-m3" in chunks[0].text
            assert chunks[0].locations[0].slide == 1
    assert warnings == [("scanned_or_empty_page", "2")]


def test_bad_parser_continues(cfg):
    broken = cfg.sources[0].path / "bad.pdf"
    broken.write_bytes(b"not a pdf")
    warnings = []
    assert parse(broken, lambda *args: warnings.append(args)) == []
    assert warnings[0][0] == "parser_error"


def test_chunk_windows_are_bounded_and_overlapping(cfg):
    text = " ".join(f"word{i}" for i in range(500))
    parts = windows(text, 50, 0.15)
    assert all(len(part) <= 200 for part in parts)
    assert len(parts) > 2
    assert parts[0][-15:] in parts[1]
    item = FileRecord("test", Path("large.md"), "large.md", len(text), 123)
    assert chunk([], item, "hash", cfg) == []


def test_config_rejects_remote_ollama_and_data_escape(tmp_path):
    sources = (Source("fixtures", tmp_path / "fixtures"),)
    with pytest.raises(ValueError, match="loopback"):
        Config(sources, base_url="https://example.com")
    with pytest.raises(ValueError, match="inside the project"):
        Config(sources, data_dir=Path("/outside"))
