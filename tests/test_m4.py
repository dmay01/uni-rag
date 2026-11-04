import json
import re
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from tests.test_m1 import transport
from uni_rag.cli import app
from uni_rag.evaluate import Question, admits_missing, checks, evaluate, load_questions
from uni_rag.ingest.pipeline import ingest
from uni_rag.models import Answer, Chunk, Hit, Location
from uni_rag.ollama import Ollama
from uni_rag.report import report

EXAMPLES = Path(__file__).parents[1] / "eval/questions.example.yaml"


def evaluation_transport(actions):
    def respond(request):
        if request.url.path == "/api/tags":
            return transport(request)
        payload = json.loads(request.content)
        if request.url.path == "/api/show":
            return transport(request)
        if request.url.path == "/api/generate":
            assert payload["keep_alive"] == 0
            actions.append(("unload", payload["model"]))
            return httpx.Response(200, json={"done": True})
        if request.url.path == "/api/embed":
            return transport(request)
        assert payload["think"] is False
        if "local answer evaluator" in payload["messages"][0]["content"]:
            actions.append(("judge", payload["model"]))
            text = json.dumps({"grounded": True, "correct": True, "explanation": "Local opinion"})
        else:
            actions.append(("answer", payload["model"]))
            content = payload["messages"][-1]["content"]
            question = content.rsplit("Question: ", 1)[1]
            expected = "notes.md"
            text = "Dense vectors and BM25"
            if "password" in question:
                text = "The answer is not in the documents."
            else:
                if "fusion use" in question:
                    expected, text = "lecture.pdf", "Rank positions"
                elif "speaker notes" in question:
                    expected, text = "slides.pptx", "bge-m3"
                elif "Sprachen" in question:
                    expected, text = "policy.docx", "Deutsch oder Englisch"
                identifier = re.search(r"\[(\d+)\] fixtures: " + re.escape(expected), content)
                text += f" [{identifier[1]}]." if identifier else "."
        return httpx.Response(
            200,
            text=json.dumps({"message": {"content": text}})
            + "\n"
            + json.dumps(
                {
                    "done": True,
                    "prompt_eval_count": 100,
                    "eval_count": 10,
                    "eval_duration": 1_000_000_000,
                }
            )
            + "\n",
        )

    return respond


def test_three_model_eval_warmup_unload_judge_and_report(cfg):
    actions = []
    models = ["qwen3:4b", "qwen3:14b", "qwen3:30b-a3b"]
    with Ollama(cfg, httpx.MockTransport(evaluation_transport(actions))) as client:
        ingest(cfg, ollama=client)
        result = evaluate(cfg, EXAMPLES, models, judge_model="local-judge", client=client)
    assert result["status"] == "complete" and result["completed_answers"] == 15
    for model in models:
        assert actions.count(("unload", model)) == 2
        assert actions.count(("answer", model)) == 6  # one warm-up and five measured answers
    assert actions.count(("judge", "local-judge")) == 15
    first_judge = next(i for i, action in enumerate(actions) if action[0] == "judge")
    assert all(action[0] != "answer" for action in actions[first_judge:])
    path, text = report(cfg)
    assert path == cfg.data_dir / "reports/summary.md"
    assert all(model in text for model in models)
    assert "Warm-up queries are excluded" in text
    assert "| 5 |" in text and "| 6 |" not in text
    rows = [
        json.loads(line)
        for line in (cfg.data_dir / "eval" / result["run_id"] / "results.jsonl")
        .read_text()
        .splitlines()
    ]
    missing = [row for row in rows if row["out_of_scope"]]
    assert all(row["hit_at_k"] is None and row["admits_missing_information"] for row in missing)
    assert all(row["metrics"]["energy_gross_joules"] is None for row in rows)
    assert all(row["metrics"]["think"] is False for row in rows)


def test_checks_use_actual_citations_and_mrr():
    hits = [
        Hit(Chunk(str(i), "text", "hash", [Location("fixtures", name, ".md")]), 1.0)
        for i, name in enumerate(["unrelated.md", "notes.md", "other.md"])
    ]
    question = Question("q", "question", ("notes.md",))
    answer = Answer("Correct words but wrong citation [1], invalid [99].", hits, {})
    result = checks(question, answer)
    assert result["hit_at_k"] == 1 and result["mrr"] == 0.5
    assert result["cites_expected_source"] is False
    assert result["invalid_citation_ids"] == [99]
    answer.text = "Correct citation [2]."
    assert checks(question, answer)["cites_expected_source"] is True


@pytest.mark.parametrize(
    "text",
    [
        "The documents do not contain this information.",
        "Das ist nicht in den Dokumenten angegeben.",
        "Dazu gibt es keine Informationen.",
    ],
)
def test_missing_admission_is_bilingual(text):
    assert admits_missing(text)
    assert not admits_missing("The seminar takes place on Tuesday.")


def test_question_validation_and_empty_report(cfg):
    assert len(load_questions(EXAMPLES)) == 5
    invalid = cfg.root / "invalid.yaml"
    invalid.write_text("questions:\n  - question: Test\n    expected_sources: notes.md\n")
    with pytest.raises(ValueError, match="expected_sources"):
        load_questions(invalid)
    path, text = report(cfg)
    assert path.exists() and "No completed evaluation" in text
    assert "| — |" in text
    with pytest.raises(ValueError, match="single evaluation directory"):
        report(cfg, "../escape")


def test_cli_all_commands_and_energy_choices():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "ask", "chat", "stats", "eval", "report"):
        assert runner.invoke(app, [command, "--help"]).exit_code == 0
    assert runner.invoke(app, ["ask", "question", "--energy", "invalid"]).exit_code == 2


def test_chat_history_and_mode_stay_local(cfg, monkeypatch):
    import uni_rag.cli as cli

    with Ollama(cfg, httpx.MockTransport(transport)) as client:
        ingest(cfg, ollama=client)
    monkeypatch.setattr(cli, "configuration", lambda path: cfg)
    calls = []

    def history_transport(request):
        if request.url.path == "/api/tags":
            return transport(request)
        payload = json.loads(request.content)
        if request.url.path == "/api/chat":
            calls.append(payload["messages"][-1]["content"])
        return transport(request)

    monkeypatch.setattr(
        cli, "Ollama", lambda config: Ollama(cfg, httpx.MockTransport(history_transport))
    )
    result = CliRunner().invoke(
        app, ["chat"], input="What is hybrid retrieval?\nExplain it.\n/exit\n"
    )
    assert result.exit_code == 0
    metrics = [
        json.loads(line)
        for path in (cfg.data_dir / "metrics").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert metrics[-1]["mode"] == "chat"
    assert "What is hybrid retrieval?" in calls[-1]
    assert "Vectors and BM25" in calls[-1]


def test_eval_records_failure_and_continues(cfg, monkeypatch):
    from uni_rag.energy import PowerSampler

    questions = cfg.root / "questions.yaml"
    questions.write_text(
        "questions:\n"
        "  - {id: first, question: 'What is hybrid retrieval?', "
        "expected_sources: [notes.md]}\n"
        "  - {id: broken, question: 'fail query', expected_sources: [notes.md]}\n"
        "  - {id: missing, question: 'What is the password?', expected_sources: []}\n"
    )
    base = evaluation_transport([])
    baseline_calls = []
    monkeypatch.setattr(
        PowerSampler, "measure_idle", lambda self, seconds: baseline_calls.append(seconds)
    )

    def failing_question(request):
        if request.url.path == "/api/chat" and json.loads(request.content)["messages"][-1][
            "content"
        ].endswith("Question: fail query"):
            return httpx.Response(500, json={"error": "synthetic model failure"})
        return base(request)

    with Ollama(cfg, httpx.MockTransport(failing_question)) as client:
        ingest(cfg, ollama=client)
        result = evaluate(cfg, questions, ["qwen3:4b"], client=client)
    assert result["status"] == "partial"
    assert result["completed_answers"] == 2 and result["failed_answers"] == 1
    assert baseline_calls == [cfg.baseline_seconds]
    assert "failed: 1" in report(cfg)[1]
