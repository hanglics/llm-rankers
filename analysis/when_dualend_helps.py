"""
When DualEnd Helps

Produces a paper-facing qualitative report comparing a TopDown baseline against
DualEnd on one or more TREC DL result folders. The analysis focuses on:

1. Per-query NDCG@10 deltas
2. Difficulty terciles using BM25 as the anchor
3. Top-k change categories (adds relevant docs, same-set reorderings, etc.)
4. Concrete positive and negative exemplar queries with document snippets

Usage:
    python analysis/when_dualend_helps.py \
        --result_dirs results/flan-t5-xl-dl19 results/qwen3-8b-dl19 results/qwen3-14b-dl19 \
        --topdown_name topdown_heapsort \
        --dualend_name dualend_bubblesort \
        --output_dir results/analysis/dualend_qualitative
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DATASET_MAP = {
    "dl19": {
        "qrels": "dl19-passage",
        "ir_dataset_name": "msmarco-passage/trec-dl-2019/judged",
        "bm25_run": Path("runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"),
    },
    "dl20": {
        "qrels": "dl20-passage",
        "ir_dataset_name": "msmarco-passage/trec-dl-2020/judged",
        "bm25_run": Path("runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--topdown_name", default="topdown_heapsort")
    parser.add_argument("--dualend_name", default="dualend_bubblesort")
    parser.add_argument("--output_dir", type=Path, default=Path("results/analysis/dualend_qualitative"))
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_examples", type=int, default=5)
    return parser.parse_args()


def infer_dataset_key(result_dir: Path) -> str:
    for key in DATASET_MAP:
        if key in result_dir.name:
            return key
    raise ValueError(f"Could not infer dataset key from {result_dir}")


def get_per_query_metric(run_path: Path, qrels: str, metric: str = "ndcg_cut.10") -> Dict[str, float]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyserini.eval.trec_eval",
            "-c",
            "-l",
            "2",
            "-m",
            metric,
            "-q",
            qrels,
            str(run_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    key = metric.replace(".", "_")
    per_query = {}
    for line in result.stdout.strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == key and parts[1] != "all":
            per_query[parts[1]] = float(parts[2])
    return per_query


def has_live_eval_stack() -> bool:
    return importlib.util.find_spec("pyserini") is not None


def load_run(path: Path) -> Dict[str, List[Tuple[str, int, float]]]:
    rankings: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, docid, rank, score = parts[0], parts[2], int(parts[3]), float(parts[4])
            rankings[qid].append((docid, rank, score))
    for qid in rankings:
        rankings[qid].sort(key=lambda item: item[1])
    return dict(rankings)


def load_dataset_context(ir_dataset_name: str):
    try:
        import ir_datasets  # type: ignore
    except ModuleNotFoundError:
        return None, {}, {}

    dataset = ir_datasets.load(ir_dataset_name)
    queries = {query.query_id: query.text for query in dataset.queries_iter()}
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrels[qrel.query_id][qrel.doc_id] = int(qrel.relevance)
    return dataset.docs_store(), queries, qrels


def get_doc_text(docstore, docid: str) -> str:
    if docstore is None:
        return ""
    doc = docstore.get(docid)
    text = getattr(doc, "text", "") or ""
    title = getattr(doc, "title", "") or ""
    content = f"{title} {text}".strip()
    return " ".join(content.split())


def compute_difficulty_terciles(bm25_scores: Dict[str, float]) -> Dict[str, str]:
    sorted_qids = sorted(bm25_scores, key=bm25_scores.get)
    n = len(sorted_qids)
    tercile = max(1, n // 3)
    mapping = {}
    for idx, qid in enumerate(sorted_qids):
        if idx < tercile:
            mapping[qid] = "hard"
        elif idx < 2 * tercile:
            mapping[qid] = "medium"
        else:
            mapping[qid] = "easy"
    return mapping


def doc_relevance(qrels: Dict[str, Dict[str, int]], qid: str, docid: str) -> int:
    return qrels.get(qid, {}).get(docid, 0)


def classify_change(
    qid: str,
    td_docs: List[str],
    de_docs: List[str],
    qrels: Dict[str, Dict[str, int]],
) -> str:
    if not qrels:
        if set(td_docs) == set(de_docs):
            return "same-set reordering"
        return "top-k set change"

    td_set = set(td_docs)
    de_set = set(de_docs)
    td_rel = {doc for doc in td_docs if doc_relevance(qrels, qid, doc) > 0}
    de_rel = {doc for doc in de_docs if doc_relevance(qrels, qid, doc) > 0}

    if td_set == de_set:
        return "same-set reordering"

    relevant_gain = len(de_rel - td_rel)
    relevant_loss = len(td_rel - de_rel)

    td_nonrel = {doc for doc in td_docs if doc_relevance(qrels, qid, doc) == 0}
    de_nonrel = {doc for doc in de_docs if doc_relevance(qrels, qid, doc) == 0}
    distractor_removed = len(td_nonrel - de_nonrel)
    distractor_added = len(de_nonrel - td_nonrel)

    if relevant_gain > relevant_loss:
        return "adds relevant documents"
    if relevant_loss > relevant_gain:
        return "drops relevant documents"
    if distractor_removed > distractor_added:
        return "removes distractors"
    if distractor_added > distractor_removed:
        return "adds distractors"
    return "mixed set change"


def summarize_changed_docs(
    qid: str,
    td_docs: List[str],
    de_docs: List[str],
    qrels: Dict[str, Dict[str, int]],
    docstore,
) -> Dict[str, List[Dict[str, object]]]:
    td_only = [doc for doc in td_docs if doc not in set(de_docs)]
    de_only = [doc for doc in de_docs if doc not in set(td_docs)]
    shared = [doc for doc in de_docs if doc in set(td_docs)]

    def build_records(docids: Iterable[str]) -> List[Dict[str, object]]:
        records = []
        for docid in docids:
            records.append({
                "docid": docid,
                "relevance": doc_relevance(qrels, qid, docid) if qrels else None,
                "snippet": get_doc_text(docstore, docid)[:220] if docstore is not None else "",
            })
        return records

    return {
        "topdown_only": build_records(td_only),
        "dualend_only": build_records(de_only),
        "shared": build_records(shared[:3]),
    }


def render_example(
    item: Dict[str, object],
    queries: Dict[str, str],
) -> str:
    lines = [
        f"#### {item['qid']} ({item['category']}, {item['difficulty']}, delta={item['delta']:+.4f})",
        "",
        f"Query: {queries.get(item['qid'], '(missing query text)')}",
        "",
        f"- TopDown NDCG@10: {item['topdown_ndcg']:.4f}",
        f"- DualEnd NDCG@10: {item['dualend_ndcg']:.4f}",
        f"- Top-k overlap: {item['topk_overlap']}/{item['top_k']}",
        "",
        "DualEnd-only top-k documents:",
    ]
    dualend_only = item["changed_docs"]["dualend_only"] or []
    if dualend_only:
        for record in dualend_only:
            relevance = record["relevance"]
            snippet = record["snippet"]
            lines.append(
                f"- `{record['docid']}`"
                f"{f' (rel={relevance})' if relevance is not None else ''}"
                f"{f': {snippet}' if snippet else ''}"
            )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("TopDown-only top-k documents:")
    topdown_only = item["changed_docs"]["topdown_only"] or []
    if topdown_only:
        for record in topdown_only:
            relevance = record["relevance"]
            snippet = record["snippet"]
            lines.append(
                f"- `{record['docid']}`"
                f"{f' (rel={relevance})' if relevance is not None else ''}"
                f"{f': {snippet}' if snippet else ''}"
            )
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def parse_saved_summary_artifacts(result_dir: Path) -> Tuple[Dict[str, object], str]:
    model, dataset = result_dir.name.rsplit("-", 1)
    artifact_key = f"{model}_{dataset}"
    per_query_path = Path("results/per_query_wins") / f"{artifact_key}.txt"
    difficulty_path = Path("results/query_difficulty") / f"{artifact_key}.txt"
    agreement_path = Path("results/ranking_agreement") / f"{artifact_key}.txt"

    if not (per_query_path.exists() and difficulty_path.exists() and agreement_path.exists()):
        raise FileNotFoundError(
            f"Fallback artifacts not found for {result_dir.name}. "
            f"Expected {per_query_path}, {difficulty_path}, and {agreement_path}."
        )

    per_query_text = per_query_path.read_text()
    difficulty_text = difficulty_path.read_text()
    agreement_text = agreement_path.read_text()

    td_de_match = re.search(
        r"TopDown vs DualEnd:\s+TopDown\s+NDCG@10=([0-9.]+)\s+wins\s+(\d+)\s+queries.*?\n"
        r"\s+DualEnd\s+NDCG@10=([0-9.]+)\s+wins\s+(\d+)\s+queries.*?\n"
        r"\s+Ties:\s+(\d+)\s+queries",
        per_query_text,
        re.DOTALL,
    )
    if not td_de_match:
        raise ValueError(f"Could not parse TopDown vs DualEnd block in {per_query_path}")

    difficulty_rows = re.findall(
        r"^(Easy|Medium|Hard)\s+\d+\s+[0-9.\-]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[+\-0-9.]+\s+\+\s*([0-9.]+)$",
        difficulty_text,
        re.MULTILINE,
    )
    overall_match = re.search(
        r"^Overall\s+\d+\s+[0-9.\s]*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+[+\-0-9.]+\s+\+\s*([0-9.]+)$",
        difficulty_text,
        re.MULTILINE,
    )
    agreement_match = re.search(
        r"TopDown vs DualEnd\s+([0-9.]+)\s+([0-9.]+)\s+(\w+)",
        agreement_text,
    )

    help_by_difficulty = {
        difficulty.lower(): float(delta) for difficulty, delta in difficulty_rows
    }
    summary = {
        "result_dir": result_dir.name,
        "dataset": dataset,
        "n_queries": int(td_de_match.group(2)) + int(td_de_match.group(4)) + int(td_de_match.group(5)),
        "mean_topdown_ndcg": float(td_de_match.group(1)),
        "mean_dualend_ndcg": float(td_de_match.group(3)),
        "mean_delta": float(td_de_match.group(3)) - float(td_de_match.group(1)),
        "help_count": int(td_de_match.group(4)),
        "hurt_count": int(td_de_match.group(2)),
        "tie_count": int(td_de_match.group(5)),
        "help_by_category": {},
        "hurt_by_category": {},
        "help_by_difficulty": help_by_difficulty,
        "hurt_by_difficulty": {},
        "fallback": True,
        "agreement_overlap": float(agreement_match.group(1)) if agreement_match else None,
        "agreement_tau": float(agreement_match.group(2)) if agreement_match else None,
        "agreement_level": agreement_match.group(3) if agreement_match else None,
        "overall_de_td": float(overall_match.group(4)) if overall_match else None,
    }

    lines = [
        f"# When DualEnd Helps: {result_dir.name}",
        "",
        "- Context source available: no (saved-artifact fallback)",
        "- Query-level exemplars unavailable in this environment because `pyserini` / `ir_datasets` are missing.",
        f"- Mean TopDown NDCG@10: {summary['mean_topdown_ndcg']:.4f}",
        f"- Mean DualEnd NDCG@10: {summary['mean_dualend_ndcg']:.4f}",
        f"- Mean delta (DualEnd - TopDown): {summary['mean_delta']:+.4f}",
        f"- Help / Hurt / Tie: {summary['help_count']} / {summary['hurt_count']} / {summary['tie_count']}",
        "",
        "## Difficulty Summary",
        "",
        "| Difficulty | DualEnd - TopDown |",
        "|---|---:|",
    ]
    for difficulty in ["easy", "medium", "hard"]:
        value = help_by_difficulty.get(difficulty)
        lines.append(f"| {difficulty} | {value:+.4f} |" if value is not None else f"| {difficulty} | - |")

    lines.extend([
        "",
        "## Agreement Summary",
        "",
        f"- Overlap@10: {summary['agreement_overlap']:.1f}" if summary["agreement_overlap"] is not None else "- Overlap@10: -",
        f"- Kendall tau: {summary['agreement_tau']:.4f}" if summary["agreement_tau"] is not None else "- Kendall tau: -",
        f"- Agreement label: {summary['agreement_level']}" if summary["agreement_level"] else "- Agreement label: -",
        "",
    ])
    return summary, "\n".join(lines)


def analyze_result_dir(
    result_dir: Path,
    topdown_name: str,
    dualend_name: str,
    top_k: int,
    num_examples: int,
) -> Tuple[Dict[str, object], str]:
    if not has_live_eval_stack():
        return parse_saved_summary_artifacts(result_dir)

    dataset_key = infer_dataset_key(result_dir)
    dataset_info = DATASET_MAP[dataset_key]
    docstore, queries, qrels = load_dataset_context(dataset_info["ir_dataset_name"])

    topdown_run = result_dir / f"{topdown_name}.txt"
    dualend_run = result_dir / f"{dualend_name}.txt"
    bm25_run = dataset_info["bm25_run"]

    td_scores = get_per_query_metric(topdown_run, dataset_info["qrels"])
    de_scores = get_per_query_metric(dualend_run, dataset_info["qrels"])
    bm25_scores = get_per_query_metric(bm25_run, dataset_info["qrels"])
    difficulties = compute_difficulty_terciles(bm25_scores)

    td_rankings = load_run(topdown_run)
    de_rankings = load_run(dualend_run)
    common_qids = sorted(set(td_scores) & set(de_scores) & set(td_rankings) & set(de_rankings))

    entries: List[Dict[str, object]] = []
    for qid in common_qids:
        td_docs = [docid for docid, _, _ in td_rankings[qid][:top_k]]
        de_docs = [docid for docid, _, _ in de_rankings[qid][:top_k]]
        category = classify_change(qid, td_docs, de_docs, qrels)
        overlap = len(set(td_docs) & set(de_docs))
        entry = {
            "qid": qid,
            "category": category,
            "difficulty": difficulties.get(qid, "unknown"),
            "topdown_ndcg": td_scores[qid],
            "dualend_ndcg": de_scores[qid],
            "delta": de_scores[qid] - td_scores[qid],
            "topk_overlap": overlap,
            "top_k": top_k,
            "changed_docs": summarize_changed_docs(qid, td_docs, de_docs, qrels, docstore),
        }
        entries.append(entry)

    helps = [entry for entry in entries if entry["delta"] > 0]
    hurts = [entry for entry in entries if entry["delta"] < 0]
    ties = [entry for entry in entries if entry["delta"] == 0]

    summary = {
        "result_dir": result_dir.name,
        "dataset": dataset_key,
        "n_queries": len(entries),
        "mean_topdown_ndcg": sum(entry["topdown_ndcg"] for entry in entries) / len(entries),
        "mean_dualend_ndcg": sum(entry["dualend_ndcg"] for entry in entries) / len(entries),
        "mean_delta": sum(entry["delta"] for entry in entries) / len(entries),
        "help_count": len(helps),
        "hurt_count": len(hurts),
        "tie_count": len(ties),
        "help_by_category": dict(Counter(entry["category"] for entry in helps)),
        "hurt_by_category": dict(Counter(entry["category"] for entry in hurts)),
        "help_by_difficulty": dict(Counter(entry["difficulty"] for entry in helps)),
        "hurt_by_difficulty": dict(Counter(entry["difficulty"] for entry in hurts)),
    }

    helps_sorted = sorted(helps, key=lambda entry: entry["delta"], reverse=True)[:num_examples]
    hurts_sorted = sorted(hurts, key=lambda entry: entry["delta"])[:num_examples]

    lines = [
        f"# When DualEnd Helps: {result_dir.name}",
        "",
        f"- Context source available: {'yes' if docstore is not None else 'no (structure-only fallback)'}",
        f"- Queries: {summary['n_queries']}",
        f"- Mean TopDown NDCG@10: {summary['mean_topdown_ndcg']:.4f}",
        f"- Mean DualEnd NDCG@10: {summary['mean_dualend_ndcg']:.4f}",
        f"- Mean delta (DualEnd - TopDown): {summary['mean_delta']:+.4f}",
        f"- Help / Hurt / Tie: {summary['help_count']} / {summary['hurt_count']} / {summary['tie_count']}",
        "",
        "## Help Categories",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for category, count in sorted(summary["help_by_category"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {category} | {count} |")

    lines.extend([
        "",
        "## Hurt Categories",
        "",
        "| Category | Count |",
        "|---|---:|",
    ])
    for category, count in sorted(summary["hurt_by_category"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {category} | {count} |")

    lines.extend([
        "",
        "## Difficulty Breakdown",
        "",
        "| Difficulty | Help | Hurt |",
        "|---|---:|---:|",
    ])
    for difficulty in ["easy", "medium", "hard"]:
        lines.append(
            f"| {difficulty} | {summary['help_by_difficulty'].get(difficulty, 0)} | "
            f"{summary['hurt_by_difficulty'].get(difficulty, 0)} |"
        )

    lines.extend([
        "",
        "## Strongest Positive Examples",
        "",
    ])
    for item in helps_sorted:
        lines.append(render_example(item, queries))

    lines.extend([
        "",
        "## Strongest Negative Examples",
        "",
    ])
    for item in hurts_sorted:
        lines.append(render_example(item, queries))

    report = "\n".join(lines)
    return summary, report


def main() -> None:
    args = parse_args()

    combined_summary = []
    combined_lines = ["# When DualEnd Helps: Combined Summary", ""]

    for result_dir in args.result_dirs:
        output_dir = Path(f'{args.output_dir}/{result_dir.name}')
        output_dir.mkdir(parents=True, exist_ok=True)
        summary, report = analyze_result_dir(
            result_dir=result_dir,
            topdown_name=args.topdown_name,
            dualend_name=args.dualend_name,
            top_k=args.top_k,
            num_examples=args.num_examples,
        )
        combined_summary.append(summary)
        (output_dir / f"{result_dir.name}_{args.topdown_name}_vs_{args.dualend_name}.md").write_text(report)
        combined_lines.extend([
            f"## {summary['result_dir']}",
            "",
            f"- Mean delta: {summary['mean_delta']:+.4f}",
            f"- Help / Hurt / Tie: {summary['help_count']} / {summary['hurt_count']} / {summary['tie_count']}",
            f"- Help categories: {json.dumps(summary['help_by_category'], sort_keys=True)}",
            f"- Hurt categories: {json.dumps(summary['hurt_by_category'], sort_keys=True)}",
            "",
        ])

    (output_dir / "WHEN_DUALEND_HELPS_SUMMARY.md").write_text("\n".join(combined_lines))
    with (output_dir / "when_dualend_helps_summary.json").open("w") as handle:
        json.dump(combined_summary, handle, indent=2)
    print(f"Wrote DualEnd qualitative analysis to {output_dir}")


if __name__ == "__main__":
    main()
