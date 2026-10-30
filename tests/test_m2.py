import json
from dataclasses import replace

import httpx
import pytest

from tests.test_m1 import transport
from uni_rag.generate import ask, source_lines
from uni_rag.ingest.manifest import active_files
from uni_rag.ingest.pipeline import ingest
from uni_rag.ollama import Ollama
from uni_rag.retrieve import active_chunks, retrieve
from uni_rag.store.vectors import VectorStore


def test_incremental_duplicate_delete_and_readd(cfg, monkeypatch):
    root = cfg.sources[0].path
    original = root / "notes.md"
    duplicate = root / "copy.md"
    duplicate.write_bytes(original.read_bytes())
    with Ollama(cfg, httpx.MockTransport(transport)) as client:
        first = ingest(cfg, ollama=client)
        assert first["duplicate_files"] == 1
        store = VectorStore(cfg.data_dir / "index")
        chunks, _ = active_chunks(cfg, store)
        notes = [c for c in chunks if any(loc.path == "notes.md" for loc in c.locations)]
        assert all({loc.path for loc in c.locations} == {"notes.md", "copy.md"} for c in notes)
        assert len(active_files(cfg.data_dir)) == 7

        def no_embedding(texts):
            raise AssertionError("Unchanged documents were re-embedded")

        with monkeypatch.context() as context:
            context.setattr(client, "embed", no_embedding)
            assert ingest(cfg, ollama=client)["files_unchanged"] == 7
            duplicate.unlink()
            result = ingest(cfg, ollama=client)
            assert result["files_removed"] == 1
        original.write_text("# Changed\nCompletely new teaching material.")
        assert ingest(cfg, ollama=client)["files_indexed"] == 1
        notes = [
            c
            for c in VectorStore(cfg.data_dir / "index").chunks()
            if any(loc.path == "notes.md" for loc in c.locations)
        ]
        assert len(notes) == 1 and "Completely new" in notes[0].text
        saved = original.read_text()
        original.unlink()
        assert ingest(cfg, ollama=client)["files_removed"] == 1
        original.write_text(saved)
        assert ingest(cfg, ollama=client)["embedding_chunks_written"] == 1


def test_resume_completed_batches_are_hidden_until_file_commits(cfg):
    cfg = replace(cfg, embedding_batch_size=1, target_tokens=16)
    for item in cfg.sources[0].path.iterdir():
        item.unlink()
    (cfg.sources[0].path / "long.txt").write_text("retrieval " * 100)
    calls = 0

    def interrupt(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return transport(request)

    with Ollama(cfg, httpx.MockTransport(interrupt)) as client:
        with pytest.raises(RuntimeError, match="simulated"):
            ingest(cfg, ollama=client)
    store = VectorStore(cfg.data_dir / "index")
    assert len(store.ids()) == 1
    assert active_chunks(cfg, store)[0] == []
    with Ollama(cfg, httpx.MockTransport(transport)) as client:
        report = ingest(cfg, ollama=client)
    assert report["embedding_chunks_written"] == report["total_chunks"] - 1


def test_filters_apply_before_rank_and_bm25_cache_reloads(cfg):
    with Ollama(cfg, httpx.MockTransport(transport)) as client:
        ingest(cfg, ollama=client)
        for _ in range(2):
            hits = retrieve("German English laboratory", cfg, client, k=1, file_type="docx")
            assert len(hits) == 1
            assert hits[0].chunk.locations[0].path == "policy.docx"
        assert retrieve("anything", cfg, client, path_prefix="missing/") == []
        hits = retrieve("MRR reciprocal rank", cfg, client, path_prefix="notes.md")
        assert all(loc.path == "notes.md" for hit in hits for loc in hit.chunk.locations)


def test_limited_and_scoped_scan_does_not_delete_other_files(cfg):
    with Ollama(cfg, httpx.MockTransport(transport)) as client:
        ingest(cfg, ollama=client)
        assert ingest(cfg, limit=1, ollama=client)["files_removed"] == 0
        nested = cfg.sources[0].path / "nested"
        nested.mkdir()
        assert ingest(cfg, only="nested", ollama=client)["files_removed"] == 0
    assert len(active_files(cfg.data_dir)) == 6


def test_parser_error_preserves_last_successful_version(cfg):
    with Ollama(cfg, httpx.MockTransport(transport)) as client:
        ingest(cfg, ollama=client)
        (cfg.sources[0].path / "lecture.pdf").write_bytes(b"corrupted pdf")
        report = ingest(cfg, ollama=client)
    assert report["skipped_by_reason"]["parser_error"] == 1
    assert len(active_files(cfg.data_dir)) == 6


def test_citations_resolve_actual_retrieved_identifiers(cfg):
    with Ollama(cfg, httpx.MockTransport(transport)) as client:
        ingest(cfg, ollama=client)
        answer = ask("What is hybrid retrieval?", cfg, client)
    assert source_lines(answer)[0].startswith("[1] fixtures:")
    answer.text = "Unsupported citation [999]."
    assert "INVALID" in source_lines(answer)[0]


def test_configuration_change_requires_fresh_index(cfg):
    with Ollama(cfg, httpx.MockTransport(transport)) as client:
        ingest(cfg, ollama=client)
        with pytest.raises(ValueError, match="settings changed"):
            ingest(replace(cfg, embedding_model="different"), ollama=client)
        with pytest.raises(ValueError, match="differs from the index"):
            retrieve("question", replace(cfg, embedding_model="different"), client)


def test_logs_and_manifest_are_local(cfg):
    ingest(cfg, dry_run=True)
    report = json.loads((cfg.data_dir / "reports/ingest.json").read_text())
    assert report["dry_run"]
