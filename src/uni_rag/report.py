"""One retrieval table and model-level generation aggregates, with coverage checks."""

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from uni_rag.config import Config
from uni_rag.eval_sources import SourceMatcher, retrieval_scores


def median(values: list) -> float | None:
    valid = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    ]
    return statistics.median(valid) if valid else None


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def read_rows(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{number}") from exc
    return rows


def select_run(config: Config, run_id: str | None = None) -> tuple[Path, dict] | None:
    candidates = sorted((config.data_dir / "eval").glob("*/run.json"), reverse=True)
    if run_id:
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run-id must be a single evaluation directory name")
        exact = config.data_dir / "eval" / run_id / "run.json"
        matches = [path for path in candidates if path.parent.name.startswith(run_id)]
        if exact.exists():
            candidates = [exact]
        elif len(matches) == 1:
            candidates = matches
        else:
            raise ValueError(f"Unknown or ambiguous evaluation run: {run_id}")
    for path in candidates:
        metadata = json.loads(path.read_text())
        if run_id or metadata["status"] in {"complete", "partial"}:
            return path, metadata
    return None


def mean_field(rows: list[dict], key: str) -> str:
    values = [r[key] for r in rows if r.get(key) is not None]
    return fmt(statistics.mean(values)) if values else "—"


def report(
    config: Config, run_id: str | None = None, compare_run_id: str | None = None
) -> tuple[Path, str]:
    selected = select_run(config, run_id)
    lines = ["# uni-rag benchmark results", ""]
    rows = []
    retrieval = []
    metadata = {}
    if selected:
        path, metadata = selected
        rows = read_rows(path.parent / "results.jsonl")
        retrieval = read_rows(path.parent / "retrieval.jsonl")
        if not retrieval:
            seen = {}
            fingerprints = defaultdict(set)
            for row in rows:
                if row.get("retrieved_sources") is not None:
                    seen.setdefault(row["question_id"], row)
                    fingerprints[row["question_id"]].add(json.dumps(row["retrieved_sources"]))
            if any(len(versions) != 1 for versions in fingerprints.values()):
                lines.extend(
                    [
                        "WARNING: saved retrieval rankings differ across models; "
                        "the retrieval table uses the first saved ranking per question.",
                        "",
                    ]
                )
            retrieval = list(seen.values())
            matcher = SourceMatcher(
                config,
                [e for r in retrieval for e in r["expected_sources"]],
                [a for r in retrieval for group in r["retrieved_sources"] for a in group],
            )
            retrieval = [
                {
                    **r,
                    **retrieval_scores(
                        r["expected_sources"],
                        r["retrieved_sources"],
                        matcher,
                        r.get("out_of_scope", False),
                    ),
                }
                for r in retrieval
            ]
        lines.extend(
            [
                f"Run: `{metadata['run_id']}`. Status: {metadata['status']}.",
                "Warm-up queries are excluded. Medians use completed measured generations; "
                "empty answers and output limits are flagged below.",
                f"Retrieval k: {metadata['k']}; questions per model: {metadata['questions']}.",
                f"Successful answers: {metadata.get('completed_answers', 0)}; "
                f"failed: {metadata.get('failed_answers', 0)}.",
                "",
            ]
        )
        if "corpus_chunks" in metadata:
            lines.extend(
                [
                    f"Corpus: {metadata['corpus_files']} files, {metadata['corpus_chunks']} "
                    f"chunks. Context window: {metadata['context_window']} tokens.",
                    "",
                ]
            )
        if metadata.get("resumed_at"):
            lines.extend(
                [
                    f"Resumed segments: {len(metadata['resumed_at'])}. Completed measurements "
                    "were retained; each resumed model segment repeats warm-up and idle "
                    "baseline measurement. Segment boundaries are recorded in run.json.",
                    "",
                ]
            )
    else:
        lines.extend(["No completed evaluation run is available.", ""])
    identities = dict(metadata.get("model_digests", {}))
    for row in rows:
        identities.setdefault(row["model"], row.get("metrics", {}).get("model_digest"))
    if identities:
        lines.extend(
            ["## Model identities", "", "| Exact model tag | SHA256 digest |", "|---|---|"]
        )
        lines.extend(
            f"| {tag} | {digest or '— (not recorded in legacy run)'} |"
            for tag, digest in identities.items()
        )
        lines.extend(
            [
                "",
                f"Output cap: {metadata.get('num_predict', '—')} tokens; "
                f"answer target: {metadata.get('answer_max_words', '—')} words.",
                "",
            ]
        )
    lines.extend(
        [
            "## Retrieval",
            "",
            "One observation per question, shared by all chat models. "
            "Out-of-scope questions are excluded from Hit@k and MRR.",
            "",
            "| Questions | Strict Hit@k | Lenient Hit@k | Strict MRR | Lenient MRR |",
            "|---|---|---|---|---|",
        ]
    )
    eligible = [r for r in retrieval if not r.get("out_of_scope") and r.get("expected_sources")]
    lines.append(
        f"| {len(eligible)} | {mean_field(eligible, 'strict_hit_at_k')} | "
        f"{mean_field(eligible, 'lenient_hit_at_k')} | {mean_field(eligible, 'mrr')} | "
        f"{mean_field(eligible, 'lenient_mrr')} |"
        if eligible
        else "| — | — | — | — | — |"
    )
    lines.extend(
        [
            "",
            "Lenient matching accepts another file with the same normalized filename "
            "stem or document-level five-word-shingle Dice similarity ≥0.95. Wrong pages "
            "within the same file do not count as copies. Matching methods are saved for audit.",
            "",
            "## Generation",
            "",
            "One row per model/think setting across all energy backends. Energy medians use "
            "only covered answers; missing readings are never replaced by zero.",
            "",
            "| Model | Think | Answers | Completion tokens | Latency (s) | TTFT (s) | "
            "Tokens/s | Gross J/answer | Above-idle J/answer | Gross J/output token | "
            "Energy n | Cite expected | Admit missing |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    groups = defaultdict(list)
    for row in rows:
        if (
            row["status"] in {"ok", "no_visible_answer"}
            and row.get("metrics", {}).get("mode") == "eval"
        ):
            groups[(row["model"], row["think"])].append(row)
    coverage_lines = []
    for (model, think), own in sorted(groups.items()):
        metrics = [row["metrics"] for row in own]

        def med(key: str, metrics: list = metrics) -> str:
            return fmt(median([m.get(key) for m in metrics]))

        covered = [m for m in metrics if m.get("energy_gross_joules") is not None]
        ratios = [
            m["energy_gross_joules"] / m["completion_tokens"]
            for m in covered
            if m.get("completion_tokens", 0) and m["completion_tokens"] > 0
        ]
        name = model.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {name} | {str(think).lower()} | {len(own)} | {med('completion_tokens')} | "
            f"{med('total_seconds')} | {med('ttft_seconds')} | {med('tokens_per_second')} | "
            f"{med('energy_gross_joules')} | {med('energy_above_baseline_joules')} | "
            f"{fmt(median(ratios))} | {len(covered)}/{len(own)} | "
            f"{mean_field(own, 'cites_expected_source')} | "
            f"{mean_field(own, 'admits_missing_information')} |"
        )
        backend_counts = dict(Counter(m.get("energy_backend", "none") for m in metrics))
        wanted = metadata.get("questions", len(own))
        incomplete = len(covered) < len(own) or len(own) < wanted
        disabled = all(m.get("energy_requested_backend") == "none" for m in metrics)
        prefix = "DISABLED" if disabled else "WARNING: INCOMPLETE" if incomplete else "COMPLETE"
        coverage_lines.append(
            f"- {prefix} energy coverage: {model}, "
            f"{len(covered)}/{len(own)} measured generations; "
            f"{len(own)}/{wanted} planned answers. Backends: {backend_counts}."
        )
        no_answer = sum(row["status"] == "no_visible_answer" for row in own)
        truncated = sum(bool(m.get("output_truncated")) for m in metrics)
        overridden = sum(bool(m.get("thinking_mode_overridden")) for m in metrics)
        repeated = sum(bool(m.get("repetition_flag")) for m in metrics)
        repeated_thought = sum(bool(m.get("thinking_repetition_flag")) for m in metrics)
        if no_answer or truncated or overridden or repeated or repeated_thought:
            coverage_lines.append(
                f"- Output flags for {model}: {no_answer} without a visible answer; "
                f"{truncated} at the token limit; {overridden} forced-thinking "
                "requests because the installed weights are thinking-only; "
                f"{repeated} visible-output and {repeated_thought} thinking repetition flags."
            )
    measured_models = {model for model, _think in groups}
    for model in metadata.get("models", []):
        if model not in measured_models:
            coverage_lines.append(
                f"- WARNING: INCOMPLETE energy coverage: {model}, 0/"
                f"{metadata.get('questions', 0)} planned answers measured."
            )
    if not groups:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | — |")
    lines.extend(
        [
            "",
            *coverage_lines,
            "",
            "J/output token is the median of paired per-answer "
            "gross joules divided by Ollama completion tokens. Energy includes the measured "
            "query window, not only decoding; completion tokens can include hidden reasoning. "
            "All generation-table numeric summaries are medians, except citation/admission rates. "
            "Citation rates use strict expected-source matching; admission checks are a "
            "bilingual lexical heuristic, not a semantic correctness judgment.",
            "",
        ]
    )
    if metadata.get("retrieval_mode") == "shared_precomputed":
        lines.extend(
            [
                "Generation latency, TTFT and energy exclude the separately computed retrieval "
                "phase. Shared retrieval time is recorded in retrieval.jsonl and each metric. "
                "Legacy runs measured retrieval inside every answer and are not directly "
                "latency/energy comparable to this generation-only protocol.",
                "",
            ]
        )
    else:
        lines.extend(["Legacy protocol: answer latency and energy include repeated retrieval.", ""])
    if metadata.get("think"):
        lines.extend(
            [
                "## Thinking experiment",
                "",
                "Token splits count returned stream logprobs, not words or character estimates. "
                "Unattributed tokens include delimiters/control tokens omitted by "
                "the thinking parser. "
                "The completion cap includes reasoning and visible output. Hit@k reuses the shared "
                "retrieval observations; citation rate checks the visible answer.",
                "",
                "| Model | Thinking tokens | Answer tokens | Unattributed tokens | Total tokens | "
                "Latency (s) | TTFT (s) | Gross J/answer | Energy n | Strict Hit@k | "
                "Lenient Hit@k | "
                "Cite expected |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for (model, _think), own in sorted(groups.items()):
            metrics = [row["metrics"] for row in own]

            def med_reason(key: str, metrics: list = metrics) -> str:
                return fmt(median([metric.get(key) for metric in metrics]))

            covered = sum(metric.get("energy_gross_joules") is not None for metric in metrics)
            lines.append(
                f"| {model} | {med_reason('thinking_tokens')} | "
                f"{med_reason('answer_tokens')} | {med_reason('unattributed_completion_tokens')} | "
                f"{med_reason('completion_tokens')} | {med_reason('total_seconds')} | "
                f"{med_reason('ttft_seconds')} | {med_reason('energy_gross_joules')} | "
                f"{covered}/{len(own)} | {mean_field(eligible, 'strict_hit_at_k')} | "
                f"{mean_field(eligible, 'lenient_hit_at_k')} | "
                f"{mean_field(own, 'cites_expected_source')} |"
            )
        lines.append("")
    if compare_run_id:
        comparison = select_run(config, compare_run_id)
        if comparison is None or (selected and comparison[0] == selected[0]):
            raise ValueError("Choose a different completed run for comparison")
        other = comparison[1]
        for key in ("questions_sha256", "k", "context_window"):
            if metadata.get(key) != other.get(key):
                raise ValueError(f"Comparison runs differ in {key}")
        if selected:
            first = read_rows(selected[0].parent / "retrieval.jsonl")
            second = read_rows(comparison[0].parent / "retrieval.jsonl")
            first_context = {r["question_id"]: r.get("retrieved_chunk_ids") for r in first}
            second_context = {r["question_id"]: r.get("retrieved_chunk_ids") for r in second}
            if first_context != second_context:
                raise ValueError("Comparison runs used different retrieved context")
        _, comparison_text = report(config, compare_run_id)
        if "## Thinking experiment" not in comparison_text:
            raise ValueError("The comparison run must be a thinking experiment")
        lines.extend(
            [
                f"## Separate reasoning run: {other['run_id']}",
                "",
                f"Status: {other['status']}; {other['questions']} questions per model.",
                "",
                "Same questions and model sizes, different weight releases and thinking mode. "
                "This is not a controlled on/off ablation of identical weights.",
                f"Resumed segments: {len(other.get('resumed_at', []))}; completed measurements "
                "are retained and resumed model segments repeat warm-up and baseline measurement.",
                "",
                comparison_text.split("## Model identities", 1)[1].split("## Retrieval", 1)[0],
                "## Thinking experiment" + comparison_text.split("## Thinking experiment", 1)[1],
                "\n".join(
                    line
                    for line in comparison_text.splitlines()
                    if "energy coverage:" in line or "Output flags for" in line
                ),
                "",
            ]
        )
    if selected and (selected[0].parent / "judgments.jsonl").exists():
        lines.append("Optional local LLM-as-judge judgments are in judgments.jsonl.\n")
    text = "\n".join(lines)
    output = config.data_dir / "reports/summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    if selected:
        (selected[0].parent / "summary.md").write_text(text, encoding="utf-8")
    return output, text
