"""Local-only diagnostics from saved retrieval results; no models or source-folder reads."""

import json
from pathlib import Path

from uni_rag.config import Config
from uni_rag.eval_sources import SourceMatcher, document, retrieval_scores, topic_similarity
from uni_rag.report import read_rows, select_run


def diagnose(config: Config, run_id: str | None = None) -> tuple[Path, str]:
    selected = select_run(config, run_id)
    if selected is None:
        raise ValueError("No evaluation run is available")
    run, metadata = selected
    rows = read_rows(run.parent / "retrieval.jsonl")
    if not rows:
        seen = set()
        for row in read_rows(run.parent / "results.jsonl"):
            if row["question_id"] not in seen and row.get("retrieved_sources") is not None:
                seen.add(row["question_id"])
                rows.append(row)
    expected = [label for row in rows for label in row["expected_sources"]]
    actual = [
        label for row in rows for group in row.get("retrieved_sources", [])[:8] for label in group
    ]
    matcher = SourceMatcher(config, expected, actual)
    lines = [
        f"# Retrieval misses: {metadata['run_id']}",
        "",
        "One row per question; saved top-8 rankings, no new retrieval. Classification (b)/(c) "
        "is a lexical heuristic requiring review; it is not a model judgment. "
        "Text similarity and topic checks use the current local index.",
        "",
    ]
    diagnostics = []
    for row in rows:
        if row.get("out_of_scope") or not row["expected_sources"]:
            continue
        rankings = row.get("retrieved_sources", [])[:8]
        scores = retrieval_scores(row["expected_sources"], rankings, matcher)
        if scores["strict_hit_at_k"]:
            continue
        texts = []
        for group in rankings:
            for label in group:
                full = matcher.documents.get(document(label), "")
                texts.extend(full[i : i + 2000] for i in range(0, len(full), 1800))
        similarity, overlap = topic_similarity(
            row.get("question", ""), row.get("reference_answer") or "", texts
        )
        if scores["lenient_hit_at_k"]:
            category = "a: another version/copy"
        elif similarity >= 0.15 and overlap >= 2:
            category = "b: right topic, different document (heuristic)"
        else:
            category = "c: unrelated (heuristic)"
        item = {
            "question_id": row["question_id"],
            "expected_sources": row["expected_sources"],
            "retrieved_sources": rankings,
            "classification": category,
            "topic_cosine": similarity,
            "shared_topic_terms": overlap,
            **scores,
        }
        diagnostics.append(item)
        lines.extend(
            [
                f"## {row['question_id']}: {category}",
                "",
                "Expected: " + "; ".join(row["expected_sources"]),
                "",
                f"Lenient hit: {bool(scores['lenient_hit_at_k'])}; "
                f"topic cosine: {similarity:.3f}; shared terms: {overlap}.",
                "",
            ]
        )
        if scores["near_duplicate_evidence"]:
            lines.extend(
                [
                    "Duplicate evidence: "
                    + json.dumps(scores["near_duplicate_evidence"], ensure_ascii=False),
                    "",
                ]
            )
        lines.extend(f"{i}. " + "; ".join(group) for i, group in enumerate(rankings, 1))
        lines.append("")
    lines.extend([f"Strict misses: {len(diagnostics)}.", ""])
    output = config.data_dir / "reports" / f"eval-misses-{metadata['run_id']}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    output.write_text(text, encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return output, text
