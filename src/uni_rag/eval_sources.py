"""Strict location matching and explicit, document-level near-duplicate evidence."""

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import lancedb

from uni_rag.config import Config
from uni_rag.ingest.manifest import active_files


def document(label: str) -> str:
    return re.sub(r", (?:page|slide) \d+(?:[–-]\d+)?$", "", label)


def normalized_stem(label: str) -> str:
    name = Path(document(label).split(": ", 1)[-1]).stem
    name = unicodedata.normalize("NFKD", name).casefold()
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"^(?:copy of|kopie von)\s+", "", name)
    name = re.sub(
        r"(?:[ _.-]+(?:copy|kopie|v\d+(?:\.\d+)*|version[ _-]*\d+)|\s*\(\d+\))$", "", name
    )
    return re.sub(r"[^\w]+", "", name)


def strict_match(expected: str, retrieved: str) -> bool:
    match = re.search(r", (page|slide) (\d+)(?:[–-](\d+))?$", expected)
    if not match:
        return expected.casefold() in retrieved.casefold()
    actual = re.search(r", (page|slide) (\d+)(?:[–-](\d+))?$", retrieved)
    if not actual or actual[1] != match[1]:
        return False
    if document(expected).casefold() not in document(retrieved).casefold():
        return False
    return int(actual[2]) <= int(match[2]) <= int(actual[3] or actual[2])


def shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
    return {tuple(words[i : i + 5]) for i in range(max(1, len(words) - 4))} if words else set()


class SourceMatcher:
    """Read only the selected indexed documents; never open original source paths."""

    def __init__(self, config: Config, expected: list[str], retrieved: list[str]):
        self.documents: dict[str, str] = {}
        self.signatures: dict[str, set] = {}
        self.pairs: dict[tuple[str, str], dict | None] = {}
        self.resolve: dict[str, list[str]] = {}
        files = active_files(config.data_dir)
        requested = {document(label) for label in expected + retrieved}
        wanted: dict[str, list[str]] = defaultdict(list)
        for item in files:
            label = f"{item['source']}: {item['path']}"
            matches = [r for r in requested if r.casefold() in label.casefold()]
            if matches:
                wanted[item["sha1"]].append(label)
                for request in matches:
                    self.resolve.setdefault(request, []).append(label)
        if not wanted:
            return
        table = lancedb.connect(str(config.data_dir / "index")).open_table("chunks")
        # One sequential scan of selected content hashes, no embeddings or index writes.
        where = " OR ".join("metadata LIKE '%" + digest + "%'" for digest in wanted)
        content: dict[str, list[str]] = defaultdict(list)
        for row in table.search().where(where).select(["text", "metadata"]).limit(None).to_list():
            digest = json.loads(row["metadata"])["content_hash"]
            if digest in wanted:
                content[digest].append(row["text"])
        for digest, labels in wanted.items():
            text = "\n".join(content[digest])
            signature = shingles(text)
            for label in labels:
                self.documents[label] = text
                self.signatures[label] = signature

    def near(self, expected: str, actual: str) -> dict | None:
        key = (document(expected), document(actual))
        if key in self.pairs:
            return self.pairs[key]
        result = None
        for candidate in self.resolve.get(key[0], [key[0]]):
            for retrieved in self.resolve.get(key[1], [key[1]]):
                # A wrong page in the very same file is not a different copy of that file.
                if candidate == retrieved:
                    continue
                if normalized_stem(candidate) == normalized_stem(retrieved):
                    result = {
                        "method": "normalized_filename_stem",
                        "expected": candidate,
                        "retrieved": retrieved,
                    }
                    break
                left, right = (
                    self.signatures.get(candidate, set()),
                    self.signatures.get(retrieved, set()),
                )
                similarity = (
                    2 * len(left & right) / (len(left) + len(right)) if left and right else 0
                )
                if similarity >= 0.95:
                    result = {
                        "method": "document_5gram_dice",
                        "similarity": similarity,
                        "expected": candidate,
                        "retrieved": retrieved,
                    }
                    break
            if result:
                break
        self.pairs[key] = result
        return result


def retrieval_scores(
    expected: list[str] | tuple[str, ...],
    rankings: list[list[str]],
    matcher: SourceMatcher | None = None,
    out_of_scope: bool = False,
) -> dict:
    strict = [any(strict_match(e, a) for e in expected for a in group) for group in rankings]
    near_evidence = []
    lenient = []
    for matched, group in zip(strict, rankings, strict=True):
        evidence = (
            next(
                (
                    found
                    for e in expected
                    for a in group
                    if matcher and (found := matcher.near(e, a))
                ),
                None,
            )
            if not matched
            else None
        )
        lenient.append(matched or evidence is not None)
        if evidence:
            near_evidence.append(evidence)
    rank = next((i for i, found in enumerate(strict, 1) if found), None)
    relaxed_rank = next((i for i, found in enumerate(lenient, 1) if found), None)
    return {
        "hit_at_k": None if out_of_scope else int(rank is not None),
        "strict_hit_at_k": None if out_of_scope else int(rank is not None),
        "lenient_hit_at_k": None if out_of_scope else int(relaxed_rank is not None),
        "mrr": None if out_of_scope else (1 / rank if rank else 0.0),
        "lenient_mrr": None if out_of_scope else (1 / relaxed_rank if relaxed_rank else 0.0),
        "near_duplicate_evidence": near_evidence,
    }


def topic_similarity(question: str, reference: str, texts: list[str]) -> tuple[float, int]:
    """Lexical cosine heuristic, explicitly not a semantic or correctness judgment."""
    stop = set(
        "the and for with that this from which what when how why does are was were has "
        "have not its into than then their oder und die der das den dem ein eine einer "
        "einem einen ist sind wird werden welche welcher warum wie was des auf für mit "
        "von als bei zu im in to of a an it is be can on or do if by each per only".split()
    )

    def terms(text):
        return Counter(
            t for t in re.findall(r"\w+", text.casefold()) if len(t) > 2 and t not in stop
        )

    left = terms(question + " " + reference)
    best, count = 0.0, 0
    for text in texts:
        right = terms(text)
        shared = left.keys() & right.keys()
        denom = (sum(n * n for n in left.values()) * sum(n * n for n in right.values())) ** 0.5
        score = sum(left[t] * right[t] for t in shared) / denom if denom else 0
        if score > best:
            best, count = score, len(shared)
    return best, count
