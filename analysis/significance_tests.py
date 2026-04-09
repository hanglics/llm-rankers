"""
Paper-facing significance tests for the main setwise experiments.

This script uses the saved per-query `.eval` files in `results/` to avoid
external dependencies such as pyserini. For each model/dataset configuration,
it compares the best method in a challenger family against the best TopDown
baseline using:

- Two-sided paired approximate randomization
- 95% paired bootstrap confidence intervals for the mean delta
- Bonferroni correction within each challenger family across the 18 configs
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np


TOPDOWN_METHODS = ("topdown_heapsort", "topdown_bubblesort")
CHALLENGER_FAMILIES = {
    "DualEnd": ("dualend_bubblesort", "dualend_selection"),
    "BottomUp": ("bottomup_heapsort", "bottomup_bubblesort"),
    "BiDir": ("bidirectional_rrf", "bidirectional_weighted_a0.7"),
}

DISPLAY_NAMES = {
    "topdown_heapsort": "TD-Heap",
    "topdown_bubblesort": "TD-Bubble",
    "dualend_bubblesort": "DE-Cocktail",
    "dualend_selection": "DE-Selection",
    "bottomup_heapsort": "BU-Heap",
    "bottomup_bubblesort": "BU-Bubble",
    "bidirectional_rrf": "BiDir-RRF",
    "bidirectional_weighted_a0.7": "BiDir-Wt(a=0.7)",
}


@dataclass
class ComparisonResult:
    family: str
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
    p_value_bonferroni: float = math.nan
    ci_low: float = math.nan
    ci_high: float = math.nan

    @property
    def direction(self) -> str:
        if self.delta > 0:
            return "improves"
        if self.delta < 0:
            return "hurts"
        return "ties"

    @property
    def significant(self) -> bool:
        return self.p_value_bonferroni < 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing model/dataset result folders.",
    )
    parser.add_argument(
        "--output-md",
        default="research_pipeline_setwise/SIGNIFICANCE_TESTS.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--output-json",
        default="research_pipeline_setwise/SIGNIFICANCE_TESTS.json",
        help="JSON report output path.",
    )
    parser.add_argument(
        "--num-randomization",
        type=int,
        default=100000,
        help="Number of approximate randomization samples per comparison.",
    )
    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=20000,
        help="Number of paired bootstrap samples per comparison.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=929,
        help="Base RNG seed for deterministic tests.",
    )
    return parser.parse_args()


def parse_results_txt(path: Path) -> dict[str, float]:
    text = path.read_text()
    lines = text.splitlines()
    start = None
    end = None
    for idx, line in enumerate(lines):
        if "Method                           nDCG@10" in line:
            start = idx + 1
        elif start is not None and "Efficiency Summary" in line:
            end = idx
            break
    if start is None or end is None:
        raise ValueError(f"Could not parse metrics table: {path}")

    metrics: dict[str, float] = {}
    for line in lines[start:end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("Method"):
            continue
        parts = stripped.split()
        if len(parts) != 6:
            continue
        method, ndcg10 = parts[0], float(parts[1])
        metrics[method] = ndcg10
    return metrics


def parse_eval_file(path: Path) -> dict[str, float]:
    per_query: dict[str, float] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        metric, qid, value = parts
        if metric == "ndcg_cut_10" and qid != "all":
            per_query[qid] = float(value)
    if not per_query:
        raise ValueError(f"No per-query ndcg_cut_10 values found in {path}")
    return per_query


def family_best(metrics: dict[str, float], methods: Iterable[str]) -> tuple[str, float]:
    available = [(method, metrics[method]) for method in methods if method in metrics]
    if not available:
        raise ValueError(f"No methods from family found: {tuple(methods)}")
    return max(available, key=lambda item: item[1])


def paired_arrays(a: dict[str, float], b: dict[str, float]) -> tuple[list[str], np.ndarray, np.ndarray]:
    common = sorted(set(a) & set(b))
    arr_a = np.array([a[qid] for qid in common], dtype=np.float64)
    arr_b = np.array([b[qid] for qid in common], dtype=np.float64)
    return common, arr_a, arr_b


def approximate_randomization(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    num_samples: int,
    seed: int,
) -> float:
    diff = arr_a - arr_b
    observed = abs(float(diff.mean()))
    if observed == 0.0:
        return 1.0

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(num_samples, diff.size))
    permuted = np.abs((signs * diff).mean(axis=1))
    more_extreme = int(np.count_nonzero(permuted >= observed - 1e-12))
    return (more_extreme + 1.0) / (num_samples + 1.0)


def paired_bootstrap_ci(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    num_samples: int,
    seed: int,
) -> tuple[float, float]:
    diff = arr_a - arr_b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diff.size, size=(num_samples, diff.size))
    samples = diff[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def bonferroni_adjust(results: list[ComparisonResult]) -> None:
    m = len(results)
    for result in results:
        result.p_value_bonferroni = min(1.0, result.p_value * m)


def format_p(p_value: float) -> str:
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def verdict(result: ComparisonResult) -> str:
    if result.significant and result.delta > 0:
        return "significant win"
    if result.significant and result.delta < 0:
        return "significant loss"
    if result.delta > 0:
        return "positive, ns"
    if result.delta < 0:
        return "negative, ns"
    return "tie"


def markdown_report(
    results_by_family: dict[str, list[ComparisonResult]],
    num_randomization: int,
    num_bootstrap: int,
) -> str:
    lines: list[str] = []
    lines.append("# Significance Tests")
    lines.append("")
    lines.append("## Method")
    lines.append(
        f"- Per-query `ndcg_cut_10` values are read from the saved `.eval` files under `results/`."
    )
    lines.append(
        "- Each comparison uses the best method within a challenger family against the best TopDown baseline for that model-dataset configuration."
    )
    lines.append(
        f"- Statistical test: two-sided paired approximate randomization with `{num_randomization:,}` samples."
    )
    lines.append(
        f"- Uncertainty: paired bootstrap 95% CI on mean delta with `{num_bootstrap:,}` resamples."
    )
    lines.append("- Multiple testing: Bonferroni correction within each family across the 18 configs.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Family | Mean delta vs best TopDown | Positive deltas | Bonferroni-significant wins | Bonferroni-significant losses |")
    lines.append("|---|---:|---:|---:|---:|")
    for family, results in results_by_family.items():
        mean_delta = np.mean([result.delta for result in results])
        positive = sum(result.delta > 0 for result in results)
        sig_wins = sum(result.significant and result.delta > 0 for result in results)
        sig_losses = sum(result.significant and result.delta < 0 for result in results)
        lines.append(
            f"| {family} | {mean_delta:+.4f} | {positive}/18 | {sig_wins} | {sig_losses} |"
        )
    lines.append("")

    for family, results in results_by_family.items():
        lines.append(f"## {family} vs Best TopDown")
        lines.append("")
        lines.append(
            "| Model | Dataset | TopDown | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for result in results:
            topdown = f"{DISPLAY_NAMES[result.baseline_method]} {result.baseline_ndcg10:.4f}"
            challenger = f"{DISPLAY_NAMES[result.challenger_method]} {result.challenger_ndcg10:.4f}"
            ci = f"[{result.ci_low:+.4f}, {result.ci_high:+.4f}]"
            wins = f"{result.wins}-{result.losses}-{result.ties}"
            lines.append(
                f"| {result.model} | {result.dataset} | {topdown} | {challenger} | "
                f"{result.delta:+.4f} | {ci} | {wins} | {format_p(result.p_value)} | "
                f"{format_p(result.p_value_bonferroni)} | {verdict(result)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_md = Path(args.output_md)
    output_json = Path(args.output_json)

    config_dirs = sorted(
        path for path in results_root.iterdir()
        if path.is_dir() and re.match(r".+-dl(19|20)$", path.name)
    )

    results_by_family: dict[str, list[ComparisonResult]] = {family: [] for family in CHALLENGER_FAMILIES}

    for config_idx, config_dir in enumerate(config_dirs):
        match = re.match(r"(.+)-dl(19|20)$", config_dir.name)
        if not match:
            continue
        model = match.group(1)
        dataset = f"DL{match.group(2)}"

        metrics = parse_results_txt(config_dir / "results.txt")
        topdown_method, topdown_ndcg10 = family_best(metrics, TOPDOWN_METHODS)

        baseline_scores = parse_eval_file(config_dir / f"{topdown_method}.eval")

        for family_idx, (family, methods) in enumerate(CHALLENGER_FAMILIES.items()):
            challenger_method, challenger_ndcg10 = family_best(metrics, methods)
            challenger_scores = parse_eval_file(config_dir / f"{challenger_method}.eval")
            common_qids, arr_challenger, arr_baseline = paired_arrays(challenger_scores, baseline_scores)
            delta = float(arr_challenger.mean() - arr_baseline.mean())
            wins = int(np.count_nonzero(arr_challenger > arr_baseline))
            losses = int(np.count_nonzero(arr_challenger < arr_baseline))
            ties = int(np.count_nonzero(arr_challenger == arr_baseline))

            seed_base = args.seed + config_idx * 100 + family_idx * 1000
            p_value = approximate_randomization(
                arr_challenger,
                arr_baseline,
                num_samples=args.num_randomization,
                seed=seed_base,
            )
            ci_low, ci_high = paired_bootstrap_ci(
                arr_challenger,
                arr_baseline,
                num_samples=args.num_bootstrap,
                seed=seed_base + 17,
            )

            results_by_family[family].append(
                ComparisonResult(
                    family=family,
                    model=model,
                    dataset=dataset,
                    baseline_method=topdown_method,
                    challenger_method=challenger_method,
                    baseline_ndcg10=topdown_ndcg10,
                    challenger_ndcg10=challenger_ndcg10,
                    delta=delta,
                    wins=wins,
                    losses=losses,
                    ties=ties,
                    n_queries=len(common_qids),
                    p_value=p_value,
                    ci_low=ci_low,
                    ci_high=ci_high,
                )
            )

    for family_results in results_by_family.values():
        bonferroni_adjust(family_results)
        family_results.sort(key=lambda item: (item.model, item.dataset))

    output_md.write_text(
        markdown_report(
            results_by_family=results_by_family,
            num_randomization=args.num_randomization,
            num_bootstrap=args.num_bootstrap,
        )
    )

    json_payload = {
        family: [asdict(result) for result in family_results]
        for family, family_results in results_by_family.items()
    }
    output_json.write_text(json.dumps(json_payload, indent=2))


if __name__ == "__main__":
    main()
