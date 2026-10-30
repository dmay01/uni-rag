"""Markdown summaries for one benchmark run; never mix warm-ups or unrelated runs."""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from uni_rag.config import Config


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
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{number}; repair the truncated line"
                ) from exc
    return rows


def report(config: Config, run_id: str | None = None) -> tuple[Path, str]:
    candidates = sorted((config.data_dir / "eval").glob("*/run.json"), reverse=True)
    if run_id:
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run-id must be a single evaluation directory name")
        candidates = [config.data_dir / "eval" / run_id / "run.json"]
        if not candidates[0].exists():
            raise ValueError(f"Unknown evaluation run: {run_id}")
    selected = None
    for path in candidates:
        metadata = json.loads(path.read_text())
        if run_id or metadata["status"] in {"complete", "partial"}:
            selected = (path, metadata)
            break
    lines = ["# uni-rag benchmark results", ""]
    groups = defaultdict(list)
    if selected:
        path, metadata = selected
        lines.extend(
            [
                f"Run: `{metadata['run_id']}`. Status: {metadata['status']}.",
                "Warm-up queries are excluded. Medians use successful measured answers.",
                "",
            ]
        )
        rows = read_rows(path.parent / "results.jsonl")
        lines.extend(
            [f"Retrieval k: {metadata['k']}; questions per model: {metadata['questions']}.", ""]
        )
        if "corpus_chunks" in metadata:
            lines.extend(
                [
                    f"Corpus: {metadata['corpus_files']} files, "
                    f"{metadata['corpus_chunks']} chunks. Context window: "
                    f"{metadata['context_window']} tokens.",
                    "",
                ]
            )
        for row in rows:
            metric = row.get("metrics", {})
            if row["status"] == "ok" and metric.get("mode") == "eval":
                groups[
                    (
                        row["model"],
                        row["think"],
                        metric.get("energy_backend", "none"),
                        metric.get("energy_sample_interval_ms"),
                    )
                ].append(row)
        lines.extend(
            [
                f"Successful answers: {metadata.get('completed_answers', 0)}; "
                f"failed: {metadata.get('failed_answers', 0)}.",
                "",
            ]
        )
    else:
        lines.extend(["No completed evaluation run is available.", ""])
    lines.extend(
        [
            "| Model | Think | Energy backend / ms | Queries | Median latency (s) | "
            "Median TTFT (s) | Tokens/s | Gross J/answer | Above-idle J/answer | "
            "Energy n | Hit@k | MRR | Cite expected | Admit missing |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for (model, think, backend, interval), rows in sorted(groups.items()):
        metrics = [row["metrics"] for row in rows]

        def med(key: str, metrics: list = metrics) -> str:
            return fmt(median([m.get(key) for m in metrics]))

        def rate(key: str, rows: list = rows) -> str:
            values = [row[key] for row in rows if row.get(key) is not None]
            return fmt(statistics.mean(values)) if values else "—"

        energy_n = sum(m.get("energy_gross_joules") is not None for m in metrics)
        escaped_model = model.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {escaped_model} | {str(think).lower()} | {backend} / {interval} | {len(rows)} | "
            f"{med('total_seconds')} | {med('ttft_seconds')} | {med('tokens_per_second')} | "
            f"{med('energy_gross_joules')} | {med('energy_above_baseline_joules')} | "
            f"{energy_n} | {rate('hit_at_k')} | {rate('mrr')} | "
            f"{rate('cites_expected_source')} | {rate('admits_missing_information')} |"
        )
    if not groups:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
    lines.extend(
        [
            "",
            "Hit@k/MRR and citation rates exclude out-of-scope questions. Admission rate "
            "uses only out-of-scope questions and is a bilingual text heuristic, not a "
            "semantic correctness score. Energy is CPU+GPU+ANE; missing values stay blank. "
            "Energy n is the number of answers with full power coverage. "
            "Different energy backends are shown separately.",
            "",
        ]
    )
    if selected and (selected[0].parent / "judgments.jsonl").exists():
        lines.append(
            "Optional local LLM-as-judge judgments are in `judgments.jsonl`; "
            "they are model opinions, separate from the deterministic checks.\n"
        )
    text = "\n".join(lines)
    output = config.data_dir / "reports/summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output, text
