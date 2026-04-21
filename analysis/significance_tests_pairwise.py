"""
Pairwise significance tests for same-sort, same-method comparisons.

Unlike `significance_tests.py` (which compares family-best challenger vs
family-best TopDown), this script compares a specific challenger method
against a specific TopDown baseline (e.g. BU-Bubble vs TD-Bubble,
DE-Cocktail vs TD-Heap). Six groupings:

    G1: TD-Bubble (baseline) vs BU-Bubble, DE-Cocktail  — bubblesort family
    G2: TD-Heap   (baseline) vs BU-Heap                  — heapsort family
    G3: TD-Bubble (baseline) vs DE-Cocktail, DE-Selection
    G4: TD-Heap   (baseline) vs DE-Cocktail, DE-Selection
    G5: TD-Bubble (baseline) vs BiDir-RRF, BiDir-Weighted(a=0.7)
    G6: TD-Heap   (baseline) vs BiDir-RRF, BiDir-Weighted(a=0.7)

For each grouping, runs per-(model, dataset, challenger) paired approximate
randomization + bootstrap CI, applies Bonferroni correction within
(grouping, dataset) across (9 models x |challengers|) tests.

Outputs:
    - SIGNIFICANCE_TESTS_PAIRWISE.md (human-readable report)
    - SIGNIFICANCE_TESTS_PAIRWISE.json (machine-readable artifact)
    - SIGNIFICANCE_TESTS_PAIRWISE.html (HTML fragment ready to paste into
      results-display/index.html under the existing family-best tables)
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from significance_tests import (  # type: ignore[import]
    DISPLAY_NAMES,
    approximate_randomization,
    bonferroni_adjust,
    format_p,
    paired_arrays,
    paired_bootstrap_ci,
    parse_eval_file,
    parse_results_txt,
)


MODEL_ORDER = [
    "flan-t5-large",
    "flan-t5-xl",
    "flan-t5-xxl",
    "qwen3-4b",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "qwen3.5-27b",
]
DATASETS = ("DL19", "DL20")


@dataclass
class Grouping:
    id: str
    title: str
    baseline: str
    challengers: tuple[str, ...]


GROUPINGS: tuple[Grouping, ...] = (
    Grouping(
        id="G1_bubble_family",
        title="Bubblesort family: TD-Bubble baseline vs BU-Bubble, DE-Cocktail",
        baseline="topdown_bubblesort",
        challengers=("bottomup_bubblesort", "dualend_bubblesort"),
    ),
    Grouping(
        id="G2_heap_family",
        title="Heapsort family: TD-Heap baseline vs BU-Heap",
        baseline="topdown_heapsort",
        challengers=("bottomup_heapsort",),
    ),
    Grouping(
        id="G3_dualend_vs_tdbubble",
        title="DualEnd vs TD-Bubble baseline",
        baseline="topdown_bubblesort",
        challengers=("dualend_bubblesort", "dualend_selection"),
    ),
    Grouping(
        id="G4_dualend_vs_tdheap",
        title="DualEnd vs TD-Heap baseline",
        baseline="topdown_heapsort",
        challengers=("dualend_bubblesort", "dualend_selection"),
    ),
    Grouping(
        id="G5_bidir_vs_tdbubble",
        title="Bidirectional vs TD-Bubble baseline",
        baseline="topdown_bubblesort",
        challengers=("bidirectional_rrf", "bidirectional_weighted_a0.7"),
    ),
    Grouping(
        id="G6_bidir_vs_tdheap",
        title="Bidirectional vs TD-Heap baseline",
        baseline="topdown_heapsort",
        challengers=("bidirectional_rrf", "bidirectional_weighted_a0.7"),
    ),
)


@dataclass
class PairwiseResult:
    grouping: str
    model: str
    dataset: str
    baseline_method: str
    challenger_method: str
    baseline_ndcg10: float
    challenger_ndcg10: float
    delta: float
    wins: int
    losses: int
    ties: int
    n_queries: int
    p_value: float
    p_value_bonferroni: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")

    @property
    def significant(self) -> bool:
        return self.p_value_bonferroni < 0.05

    @property
    def verdict(self) -> str:
        if self.significant and self.delta > 0:
            return "sig win"
        if self.significant and self.delta < 0:
            return "sig loss"
        if self.delta > 0:
            return "+ns"
        if self.delta < 0:
            return "-ns"
        return "tie"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--output-md",
        default="research_pipeline_setwise/SIGNIFICANCE_TESTS_PAIRWISE.md",
    )
    parser.add_argument(
        "--output-json",
        default="research_pipeline_setwise/SIGNIFICANCE_TESTS_PAIRWISE.json",
    )
    parser.add_argument(
        "--output-html",
        default="results-display/_pairwise_tables_fragment.html",
        help="HTML fragment containing the 12 comparison tables, ready to paste into index.html.",
    )
    parser.add_argument("--num-randomization", type=int, default=100000)
    parser.add_argument("--num-bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=929)
    return parser.parse_args()


def discover_configs(results_root: Path) -> dict[tuple[str, str], Path]:
    """Map (model, dataset) -> results directory."""
    configs: dict[tuple[str, str], Path] = {}
    for path in results_root.iterdir():
        if not path.is_dir():
            continue
        match = re.match(r"(.+)-dl(19|20)$", path.name)
        if not match:
            continue
        model = match.group(1)
        dataset = f"DL{match.group(2)}"
        configs[(model, dataset)] = path
    return configs


def run_comparison(
    config_dir: Path,
    baseline: str,
    challenger: str,
    metrics: dict[str, float],
    num_randomization: int,
    num_bootstrap: int,
    seed: int,
    model: str,
    dataset: str,
    grouping_id: str,
) -> PairwiseResult:
    baseline_scores = parse_eval_file(config_dir / f"{baseline}.eval")
    challenger_scores = parse_eval_file(config_dir / f"{challenger}.eval")
    common, arr_challenger, arr_baseline = paired_arrays(
        challenger_scores, baseline_scores
    )
    delta = float(arr_challenger.mean() - arr_baseline.mean())
    wins = int(np.count_nonzero(arr_challenger > arr_baseline))
    losses = int(np.count_nonzero(arr_challenger < arr_baseline))
    ties = int(np.count_nonzero(arr_challenger == arr_baseline))
    p_value = approximate_randomization(
        arr_challenger,
        arr_baseline,
        num_samples=num_randomization,
        seed=seed,
    )
    ci_low, ci_high = paired_bootstrap_ci(
        arr_challenger,
        arr_baseline,
        num_samples=num_bootstrap,
        seed=seed + 17,
    )
    return PairwiseResult(
        grouping=grouping_id,
        model=model,
        dataset=dataset,
        baseline_method=baseline,
        challenger_method=challenger,
        baseline_ndcg10=metrics[baseline],
        challenger_ndcg10=metrics[challenger],
        delta=delta,
        wins=wins,
        losses=losses,
        ties=ties,
        n_queries=len(common),
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def build_results(
    configs: dict[tuple[str, str], Path],
    args: argparse.Namespace,
) -> dict[tuple[str, str], list[PairwiseResult]]:
    """results_by[(grouping_id, dataset)] -> list[PairwiseResult]"""
    results_by: dict[tuple[str, str], list[PairwiseResult]] = {}
    metrics_cache: dict[tuple[str, str], dict[str, float]] = {}
    for (model, dataset), config_dir in configs.items():
        metrics_cache[(model, dataset)] = parse_results_txt(config_dir / "results.txt")

    for g_idx, grouping in enumerate(GROUPINGS):
        for dataset in DATASETS:
            bucket: list[PairwiseResult] = []
            for m_idx, model in enumerate(MODEL_ORDER):
                config_key = (model, dataset)
                if config_key not in configs:
                    continue
                config_dir = configs[config_key]
                metrics = metrics_cache[config_key]
                for c_idx, challenger in enumerate(grouping.challengers):
                    if challenger not in metrics or grouping.baseline not in metrics:
                        continue
                    seed = args.seed + g_idx * 10_000 + m_idx * 100 + c_idx
                    result = run_comparison(
                        config_dir=config_dir,
                        baseline=grouping.baseline,
                        challenger=challenger,
                        metrics=metrics,
                        num_randomization=args.num_randomization,
                        num_bootstrap=args.num_bootstrap,
                        seed=seed,
                        model=model,
                        dataset=dataset,
                        grouping_id=grouping.id,
                    )
                    bucket.append(result)
            bonferroni_adjust(bucket)  # type: ignore[arg-type]
            results_by[(grouping.id, dataset)] = bucket
    return results_by


def markdown_report(
    results_by: dict[tuple[str, str], list[PairwiseResult]],
    num_randomization: int,
    num_bootstrap: int,
) -> str:
    lines: list[str] = []
    lines.append("# Significance Tests — Pairwise (same-sort comparisons)")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        f"- Per-query `ndcg_cut_10` values are read from the saved `.eval` files under `results/`."
    )
    lines.append(
        "- Each comparison uses a **specific** TopDown baseline against a **specific** challenger method "
        "(no family-best reduction)."
    )
    lines.append(
        f"- Statistical test: two-sided paired approximate randomization with `{num_randomization:,}` samples."
    )
    lines.append(
        f"- Uncertainty: paired bootstrap 95% CI on mean delta with `{num_bootstrap:,}` resamples."
    )
    lines.append(
        "- Multiple testing: Bonferroni correction **per (grouping, dataset)** across "
        "(9 models × |challengers|) tests. Each table is its own hypothesis family."
    )
    lines.append("")

    lines.append("## Summary (per grouping, per dataset)")
    lines.append("")
    lines.append(
        "| Grouping | Dataset | Mean delta | Positive / total | Bonferroni wins | Bonferroni losses |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for grouping in GROUPINGS:
        for dataset in DATASETS:
            bucket = results_by.get((grouping.id, dataset), [])
            if not bucket:
                continue
            mean_delta = float(np.mean([r.delta for r in bucket]))
            positive = sum(r.delta > 0 for r in bucket)
            sig_wins = sum(r.significant and r.delta > 0 for r in bucket)
            sig_losses = sum(r.significant and r.delta < 0 for r in bucket)
            total = len(bucket)
            lines.append(
                f"| {grouping.title} | {dataset} | {mean_delta:+.4f} | "
                f"{positive}/{total} | {sig_wins} | {sig_losses} |"
            )
    lines.append("")

    for grouping in GROUPINGS:
        for dataset in DATASETS:
            bucket = results_by.get((grouping.id, dataset), [])
            if not bucket:
                continue
            lines.append(f"## {grouping.title} — {dataset}")
            lines.append("")
            lines.append(
                "| Model | Baseline | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |"
            )
            lines.append(
                "|---|---:|---:|---:|---:|---:|---:|---:|---|"
            )
            for r in bucket:
                baseline = f"{DISPLAY_NAMES[r.baseline_method]} {r.baseline_ndcg10:.4f}"
                challenger = f"{DISPLAY_NAMES[r.challenger_method]} {r.challenger_ndcg10:.4f}"
                ci = f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
                wlt = f"{r.wins}-{r.losses}-{r.ties}"
                lines.append(
                    f"| {r.model} | {baseline} | {challenger} | "
                    f"{r.delta:+.4f} | {ci} | {wlt} | "
                    f"{format_p(r.p_value)} | {format_p(r.p_value_bonferroni)} | {r.verdict} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


_SIG_CSS_CLASS = {
    "sig win": "sig-win",
    "sig loss": "sig-loss",
    "+ns": "sig-pos-ns",
    "-ns": "sig-neg-ns",
    "tie": "sig-neg-ns",
}


def html_fragment(
    results_by: dict[tuple[str, str], list[PairwiseResult]],
) -> str:
    """Emit HTML fragment with 12 comparison-table cards, ready to paste
    into `results-display/index.html` under the existing family-best tables."""

    def cell(r: PairwiseResult) -> str:
        cls = _SIG_CSS_CLASS.get(r.verdict, "sig-neg-ns")
        return (
            '<td><div class="table-cell-stack">'
            f'<div class="cell-main">{r.challenger_ndcg10:.4f}</div>'
            f'<span class="sig-tag {cls}">{html.escape(r.verdict)}</span>'
            "</div></td>"
        )

    def baseline_cell(r: PairwiseResult | None) -> str:
        if r is None:
            return "<td>—</td>"
        return f"<td>{r.baseline_ndcg10:.4f}</td>"

    chunks: list[str] = []
    chunks.append(
        "<!-- BEGIN pairwise comparison tables (auto-generated by "
        "analysis/significance_tests_pairwise.py) -->\n"
    )
    for grouping in GROUPINGS:
        for dataset in DATASETS:
            bucket = results_by.get((grouping.id, dataset), [])
            if not bucket:
                continue
            by_model_challenger = {
                (r.model, r.challenger_method): r for r in bucket
            }

            chunks.append('<article class="table-card">')
            chunks.append('  <div class="table-head">')
            chunks.append(
                f'    <div><span class="table-kicker">Pairwise comparison — {dataset}</span>'
                f'<h3>{html.escape(grouping.title)} — {dataset}</h3></div>'
            )
            chunks.append(
                "    <p>Baseline: <code>"
                f"{html.escape(DISPLAY_NAMES[grouping.baseline])}</code>. "
                "Verdicts use paired approximate randomization (100K samples) with "
                "Bonferroni correction inside this table.</p>"
            )
            chunks.append("  </div>")
            chunks.append('  <div class="table-wrap">')
            chunks.append('    <table class="results-table">')
            chunks.append(
                "      <thead><tr><th>Method</th>"
                + "".join(f"<th>{m}</th>" for m in MODEL_ORDER)
                + "</tr></thead>"
            )
            chunks.append("      <tbody>")

            # Baseline row (raw NDCG values only — it is the reference).
            baseline_row: list[str] = [
                f'<td>{html.escape(DISPLAY_NAMES[grouping.baseline])} '
                f"<span class=\"muted\">(baseline)</span></td>"
            ]
            for model in MODEL_ORDER:
                # pick any challenger's result for this (model, dataset) to grab the baseline ndcg
                any_r = next(
                    (by_model_challenger[(model, c)] for c in grouping.challengers
                     if (model, c) in by_model_challenger),
                    None,
                )
                baseline_row.append(baseline_cell(any_r))
            chunks.append('        <tr class="row-baseline">' + "".join(baseline_row) + "</tr>")

            # Challenger rows — NDCG + sig tag.
            for challenger in grouping.challengers:
                row_class = ' class="row-highlight"' if challenger.startswith("dualend") else ""
                row: list[str] = [
                    f"<td>{html.escape(DISPLAY_NAMES[challenger])}</td>"
                ]
                for model in MODEL_ORDER:
                    r = by_model_challenger.get((model, challenger))
                    row.append(cell(r) if r is not None else "<td>—</td>")
                chunks.append(f"        <tr{row_class}>" + "".join(row) + "</tr>")

            chunks.append("      </tbody></table></div>")
            chunks.append('  <div class="sig-legend">')
            chunks.append('    <span class="sig-tag sig-win">sig win</span>')
            chunks.append('    <span class="sig-tag sig-loss">sig loss</span>')
            chunks.append('    <span class="sig-tag sig-pos-ns">+ns</span>')
            chunks.append('    <span class="sig-tag sig-neg-ns">-ns</span>')
            chunks.append("  </div>")
            chunks.append("</article>\n")
    chunks.append("<!-- END pairwise comparison tables -->\n")
    return "\n".join(chunks)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    configs = discover_configs(results_root)

    results_by = build_results(configs, args)

    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_html = Path(args.output_html)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    output_md.write_text(markdown_report(results_by, args.num_randomization, args.num_bootstrap))

    json_payload = {
        f"{grouping_id}__{dataset}": [asdict(r) for r in bucket]
        for (grouping_id, dataset), bucket in results_by.items()
    }
    output_json.write_text(json.dumps(json_payload, indent=2))

    output_html.write_text(html_fragment(results_by))

    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
