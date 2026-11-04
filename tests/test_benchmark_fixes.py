import io
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from tests.test_m1 import transport
from tests.test_m4 import EXAMPLES, evaluation_transport
from uni_rag.answer_quality import VisibleText, quality
from uni_rag.energy import PowerSampler, Sample
from uni_rag.eval_misses import diagnose
from uni_rag.eval_sources import SourceMatcher, normalized_stem, retrieval_scores, strict_match
from uni_rag.evaluate import evaluate
from uni_rag.generate import ask
from uni_rag.ingest.pipeline import ingest
from uni_rag.ollama import Ollama
from uni_rag.report import report


def test_thought_suppression_handles_arbitrary_stream_boundaries():
    text = VisibleText()
    parts = ["<th", "ink>private thoughts", "</thi", "nk>Answer ", "<think>more", "</think> [1]."]
    assert "".join(text.feed(part) for part in parts) == "Answer  [1]."
    assert text.think_tags_seen
    hidden = VisibleText()
    assert hidden.feed("<think>unfinished", final=True) == ""
    assert quality("one two three four five six seven eight nine ten eleven twelve " * 5)[
        "repetition_flag"
    ]


def test_token_limit_and_no_think_soft_switch_are_sent(cfg):
    def mock(request):
        if request.url.path == "/api/tags":
            return transport(request)
        payload = json.loads(request.content)
        if request.url.path == "/api/show":
            return transport(request)
        assert payload["options"]["num_predict"] == 77
        assert payload["think"] is False
        assert payload["messages"][0]["content"].endswith("/no_think")
        return httpx.Response(200, text='{"done":true}\n')

    with Ollama(replace(cfg, num_predict=77), httpx.MockTransport(mock)) as client:
        list(
            client.chat(
                [{"role": "system", "content": "system"}, {"role": "user", "content": "test"}],
                "qwen3:4b",
            )
        )
    with pytest.raises(ValueError, match="num_predict"):
        replace(cfg, num_predict=-1)


def test_bounded_sampling_jitter_and_per_query_warnings(cfg, monkeypatch):
    monkeypatch.setattr(PowerSampler, "_start_macmon", lambda self: None)
    with PowerSampler(cfg, "macmon") as sampler:
        sampler.samples.extend([Sample(0, 2), Sample(0.2, 2), Sample(0.7, 2), Sample(0.9, 2)])
        good = sampler.measure(0.1, 0.8)
        assert good["energy_gross_joules"] == pytest.approx(1.4)
        assert good["energy_coverage"]["macmon"]["max_gap_ms"] == pytest.approx(500)
        bad = sampler.measure(0.1, 1.1)
        assert bad["energy_gross_joules"] is None
        assert bad["energy_coverage"]["macmon"]["reason"] == "missing_end"
        assert sampler.measure(0.1, 0.8)["energy_warnings"] == []
    with PowerSampler(replace(cfg, max_sample_gap_ms=350), "macmon") as sampler:
        sampler.samples.extend([Sample(0, 2), Sample(0.5, 2), Sample(1, 2)])
        bad = sampler.measure(0.1, 0.9)
        assert bad["energy_gross_joules"] is None
        assert bad["energy_coverage"]["macmon"]["reason"] == "sample_gap"


def test_invalid_sample_is_skipped_and_reader_continues(cfg, monkeypatch):
    monkeypatch.setattr(PowerSampler, "_start_macmon", lambda self: None)
    valid = {
        "timestamp": "2026-11-04T10:00:00+00:00",
        "cpu_power": 1,
        "gpu_power": 2,
        "ane_power": 0,
    }
    process = SimpleNamespace(
        stdout=io.StringIO("bad-json\n" + json.dumps(valid) + "\n"), poll=lambda: 1
    )
    with PowerSampler(cfg, "macmon") as sampler:
        sampler._read(process)
        assert len(sampler.samples) == 1
        assert sampler.failed
        assert any("invalid" in warning for warning in sampler.warnings)


def test_stalled_sampler_is_restarted_between_queries(cfg, monkeypatch):
    starts = []
    monkeypatch.setattr(PowerSampler, "_start_macmon", lambda self: starts.append(1))
    with PowerSampler(cfg, "auto") as sampler:
        sampler.failed = True
        sampler.prepare()
        assert len(starts) == 2 and sampler.restarts == 1


def test_duplicate_rules_use_whole_documents_and_location_boundaries(cfg):
    root = cfg.sources[0].path
    (root / "renamed-identical.md").write_bytes((root / "notes.md").read_bytes())
    (root / "notes v2.md").write_text("Completely different content about calendars.")
    with Ollama(cfg, httpx.MockTransport(transport)) as client:
        ingest(cfg, ollama=client)
    expected = "fixtures: notes.md"
    matcher = SourceMatcher(
        cfg, [expected], ["fixtures: renamed-identical.md", "fixtures: notes v2.md"]
    )
    assert (
        matcher.near(expected, "fixtures: renamed-identical.md")["method"] == "document_5gram_dice"
    )
    assert matcher.near(expected, "fixtures: notes v2.md")["method"] == "normalized_filename_stem"
    rankings = [["fixtures: renamed-identical.md"]]
    scores = retrieval_scores([expected], rankings, matcher)
    assert scores["strict_hit_at_k"] == 0 and scores["lenient_hit_at_k"] == 1
    assert strict_match("file.pdf, page 1", "file.pdf, page 10") is False
    assert strict_match("deck.pptx, slide 4", "deck.pptx, slide 4–5")
    assert not matcher.near("fixtures: notes.md, page 1", "fixtures: notes.md, page 2")
    assert normalized_stem("Copy of Model v2.pdf") == normalized_stem("Model.pdf")


def test_shared_retrieval_report_aggregates_backends_and_flags_coverage(cfg):
    with Ollama(cfg, httpx.MockTransport(evaluation_transport([]))) as client:
        ingest(cfg, ollama=client)
        result = evaluate(cfg, EXAMPLES, ["qwen3:4b"], client=client)
    run = cfg.data_dir / "eval" / result["run_id"]
    retrieval = [json.loads(line) for line in (run / "retrieval.jsonl").read_text().splitlines()]
    assert len(retrieval) == 5
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    for i, row in enumerate(rows):
        metric = row["metrics"]
        assert metric["num_predict"] == 512 and metric["answer_max_words"] == 150
        assert metric["retrieval_mode"] == "shared_precomputed"
        metric.update(
            energy_requested_backend="auto",
            energy_backend="macmon" if i < 3 else "none",
            energy_gross_joules=10 if i < 3 else None,
        )
    (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    text = report(cfg)[1]
    assert text.split("## Generation", 1)[1].count("| qwen3:4b |") == 1
    assert "WARNING: INCOMPLETE" in text and "3/5" in text
    assert "Gross J/output token" in text and "Strict Hit@k" in text
    assert "## Retrieval" in text and "## Generation" in text
    path, misses = diagnose(cfg, result["run_id"])
    assert path.exists() and "saved top-8 rankings" in misses


@pytest.mark.parametrize("visible", ["", "\n\n"])
def test_thinking_only_identity_and_budget_exhaustion_are_honest(cfg, visible):
    def mock(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {"general.finetune": "Thinking"}})
        if request.url.path == "/api/chat":
            assert json.loads(request.content)["think"] is True
            return httpx.Response(
                200,
                text='{"message":{"thinking":"hidden reasoning"}}\n'
                + json.dumps({"message": {"content": visible}})
                + "\n"
                + '{"done":true,"done_reason":"length","eval_count":512}\n',
            )
        return transport(request)

    with Ollama(cfg, httpx.MockTransport(mock)) as client:
        ingest(cfg, ollama=client)
        answer = ask("What is hybrid retrieval?", cfg, client, think=False)
    assert not answer.text.strip()
    assert answer.metrics["think"] is True and answer.metrics["think_requested"] is False
    assert answer.metrics["thinking_mode_overridden"] is True
    assert answer.metrics["status"] == "no_visible_answer"
    assert answer.metrics["output_truncated"] is True


def test_token_splits_are_from_logprobs_and_digest_is_logged(cfg):
    def mock(request):
        if request.url.path != "/api/chat":
            return transport(request)
        assert json.loads(request.content)["logprobs"] is True
        return httpx.Response(
            200,
            text='{"message":{"thinking":"hidden"},"logprobs":[{},{}]}\n'
            '{"message":{"content":"Answer [1]."},"logprobs":[{},{},{}]}\n'
            '{"done":true,"eval_count":8}\n',
        )

    with Ollama(cfg, httpx.MockTransport(mock)) as client:
        ingest(cfg, ollama=client)
        answer = ask("What is hybrid retrieval?", cfg, client, think=True)
    metric = answer.metrics
    assert metric["thinking_tokens"] == 2 and metric["answer_tokens"] == 3
    assert metric["unattributed_completion_tokens"] == 3
    assert (
        metric["thinking_tokens"]
        + metric["answer_tokens"]
        + metric["unattributed_completion_tokens"]
        == metric["completion_tokens"]
    )
    assert metric["model_digest"] == "a" * 64
    assert metric["token_split_method"] == "ollama_stream_logprobs"
    assert "hidden" not in answer.text


def test_reasoning_comparison_report_has_separate_table_and_provenance(cfg):
    from uni_rag.ingest.pipeline import write_json

    with Ollama(cfg, httpx.MockTransport(evaluation_transport([]))) as client:
        ingest(cfg, ollama=client)
        main = evaluate(cfg, EXAMPLES, ["qwen3:4b"], client=client)
    source = cfg.data_dir / "eval" / main["run_id"]
    target = cfg.data_dir / "eval" / "reasoning-fixture"
    target.mkdir()
    meta = json.loads((source / "run.json").read_text())
    meta.update(run_id="reasoning-fixture", think=True, num_predict=4096)
    write_json(target / "run.json", meta)
    (target / "retrieval.jsonl").write_text((source / "retrieval.jsonl").read_text())
    rows = [json.loads(line) for line in (source / "results.jsonl").read_text().splitlines()]
    for row in rows:
        row["think"] = True
        row["metrics"].update(
            think=True,
            thinking_tokens=100,
            answer_tokens=10,
            unattributed_completion_tokens=3,
            completion_tokens=113,
        )
    (target / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    text = report(cfg, main["run_id"], "reasoning-fixture")[1]
    assert "## Separate reasoning run" in text
    assert "| Thinking tokens | Answer tokens |" in text
    assert "100.000 | 10.000 | 3.000 | 113.000" in text
    assert "a" * 64 in text


def test_miss_diagnostic_lists_saved_rankings_and_all_three_classes(cfg):
    from uni_rag.ingest.pipeline import write_json

    root = cfg.sources[0].path
    (root / "renamed-identical.md").write_bytes((root / "notes.md").read_bytes())
    with Ollama(cfg, httpx.MockTransport(transport)) as client:
        ingest(cfg, ollama=client)
    run = cfg.data_dir / "eval" / "diagnostic-fixture"
    write_json(run / "run.json", {"run_id": "diagnostic-fixture", "status": "complete"})
    base = {"expected_sources": ["fixtures: notes.md"], "out_of_scope": False, "status": "ok"}
    rows = [
        {
            **base,
            "question_id": "copy",
            "question": "What is hybrid retrieval?",
            "retrieved_sources": [["fixtures: renamed-identical.md"]],
        },
        {
            **base,
            "question_id": "topic",
            "question": "Which languages are laboratory reports accepted in?",
            "reference_answer": "The laboratory accepts reports in German or English.",
            "retrieved_sources": [["fixtures: policy.docx"]],
        },
        {
            **base,
            "question_id": "unrelated",
            "question": "How does a river flow through a desert?",
            "reference_answer": "Water flows over dry sand.",
            "retrieved_sources": [["fixtures: schedule.txt"]],
        },
    ]
    (run / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    path, text = diagnose(cfg, "diagnostic-fixture")
    entries = json.loads(path.with_suffix(".json").read_text())
    assert [e["classification"][0] for e in entries] == ["a", "b", "c"]
    assert all(e["retrieved_sources"] for e in entries)
    assert "fixtures: notes.md" in text and "fixtures: schedule.txt" in text


def test_second_experiment_reuses_frozen_context_without_new_retrieval(cfg, monkeypatch):
    from uni_rag.retrieve import Retriever

    with Ollama(cfg, httpx.MockTransport(evaluation_transport([]))) as client:
        ingest(cfg, ollama=client)
        first = evaluate(cfg, EXAMPLES, ["qwen3:4b"], client=client)
        monkeypatch.setattr(
            Retriever, "search", lambda *args, **kwargs: pytest.fail("Retrieval was repeated")
        )
        second = evaluate(
            cfg, EXAMPLES, ["qwen3:14b"], client=client, retrieval_run_id=first["run_id"]
        )
    base = cfg.data_dir / "eval"
    assert (base / first["run_id"] / "retrieval.jsonl").read_bytes() == (
        base / second["run_id"] / "retrieval.jsonl"
    ).read_bytes()
    assert second["retrieval_reused_from"] == first["run_id"]


def test_resume_preserves_answers_and_rejects_changed_protocol(cfg, monkeypatch):
    with Ollama(cfg, httpx.MockTransport(evaluation_transport([]))) as client:
        ingest(cfg, ollama=client)
        first = evaluate(cfg, EXAMPLES, ["qwen3:4b"], client=client)
        run = cfg.data_dir / "eval" / first["run_id"]
        original = (run / "results.jsonl").read_text().splitlines(keepends=True)
        (run / "results.jsonl").write_text("".join(original[:2]))
        monkeypatch.setattr(
            "uni_rag.evaluate.Retriever.search",
            lambda *args, **kwargs: pytest.fail("Resume repeated retrieval"),
        )
        result = evaluate(cfg, EXAMPLES, ["qwen3:4b"], client=client, resume_run_id=first["run_id"])
        saved = (run / "results.jsonl").read_text().splitlines(keepends=True)
        assert saved[:2] == original[:2] and len(saved) == len(original)
        assert result["completed_answers"] == len(original)
        assert result["resumed_at"] and result["baseline_segments"]
        metadata = (run / "run.json").read_bytes()
        with pytest.raises(ValueError, match="settings or questions differ"):
            evaluate(
                replace(cfg, num_predict=123),
                EXAMPLES,
                ["qwen3:4b"],
                client=client,
                resume_run_id=first["run_id"],
            )
        assert (run / "run.json").read_bytes() == metadata
