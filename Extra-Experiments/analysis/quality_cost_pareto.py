"""
Quality-Cost Pareto Analysis

Builds paper-facing quality-cost summaries from result directories that contain
`.eval` and `.log` files. The script writes:

1. A per-run CSV (`all_points.csv`)
2. A method-average CSV (`method_means.csv`)
3. Pareto-membership CSVs for comparisons / tokens / time
4. A Markdown report suitable for paper drafting
5. Optional scatter plots if matplotlib is available

Usage:
    python analysis/quality_cost_pareto.py \
        --results_root results \
        --output_dir results/analysis/pareto
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional
    plt = None


METHOD_LABELS = {
    "topdown_heapsort": "TD-Heap",
    "topdown_bubblesort": "TD-Bubble",
    "bottomup_heapsort": "BU-Heap",
    "bottomup_bubblesort": "BU-Bubble",
    "dualend_bubblesort": "DE-Cocktail",
    "dualend_selection": "DE-Selection",
    "bidirectional_rrf": "BiDir-RRF",
    "bidirectional_weighted_a0.7": "BiDir-Weighted",
    "permvote_p2_heapsort": "PermVote(p=2)",
    "selective_dualend_heapsort": "SelDE-Heap",
    "selective_dualend_bubblesort": "SelDE-Bubble",
    "bias_aware_dualend_bubblesort": "BiasAware-DE",
    "samecall_regularized_bubblesort": "SameCall-Reg",
}

FRONTIER_COSTS = {
    "avg_comparisons": "comparisons",
    "avg_total_tokens": "total tokens",
    "avg_time_per_query": "time",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("results"),
        help="Root directory containing result folders such as qwen3-8b-dl19",
    )
    parser.add_argument(
        "--result_dirs",
        nargs="*",
        type=Path,
        default=None,
        help="Optional explicit result directories. If omitted, auto-discovers under results_root.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/analysis/pareto"),
        help="Directory to store CSV, Markdown, and plot outputs.",
    )
    parser.add_argument(
        "--include_methods",
        nargs="*",
        default=None,
        help="Optional method stems to keep (e.g., topdown_heapsort dualend_bubblesort).",
    )
    return parser.parse_args()


def discover_result_dirs(results_root: Path) -> List[Path]:
    dirs = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in {"analysis", "ablation-alpha", "ablation-nc", "ablation-pl", "parse_success",
                          "per_query_wins", "query_difficulty", "ranking_agreement"}:
            continue
        if list(child.glob("*.eval")):
            dirs.append(child)
    return dirs


def infer_model_and_dataset(dir_name: str) -> tuple[str, str]:
    match = re.match(r"(.+)-(dl\d{2})$", dir_name)
    if not match:
        return dir_name, "unknown"
    return match.group(1), match.group(2)


def safe_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_eval_file(path: Path) -> Dict[str, float]:
    metrics = {}
    with path.open() as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 3 or parts[1] != "all":
                continue
            metrics[parts[0]] = float(parts[2])
    return metrics


def parse_log_file(path: Path) -> Dict[str, float]:
    patterns = {
        "avg_comparisons": r"Avg comparisons:\s*([0-9.eE+-]+)",
        "avg_prompt_tokens": r"Avg prompt tokens:\s*([0-9.eE+-]+)",
        "avg_completion_tokens": r"Avg completion tokens:\s*([0-9.eE+-]+)",
        "avg_time_per_query": r"Avg time per query:\s*([0-9.eE+-]+)",
        "avg_dual_invocations": r"Avg dual invocations:\s*([0-9.eE+-]+)",
        "avg_single_invocations": r"Avg single invocations:\s*([0-9.eE+-]+)",
        "avg_order_robust_windows": r"Avg order-robust windows:\s*([0-9.eE+-]+)",
        "avg_extra_orderings": r"Avg extra orderings:\s*([0-9.eE+-]+)",
        "avg_regularized_worst_moves": r"Avg regularized worst moves:\s*([0-9.eE+-]+)",
    }
    text = path.read_text()
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            parsed[key] = float(match.group(1))
    parsed["avg_total_tokens"] = parsed.get("avg_prompt_tokens", 0.0) + parsed.get("avg_completion_tokens", 0.0)
    return parsed


def normalize_method_name(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def collect_points(result_dirs: Iterable[Path], include_methods: Optional[set[str]] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for result_dir in result_dirs:
        model, dataset = infer_model_and_dataset(result_dir.name)
        eval_by_method = {path.stem: parse_eval_file(path) for path in result_dir.glob("*.eval")}
        log_by_method = {path.stem: parse_log_file(path) for path in result_dir.glob("*.log")}
        methods = sorted(set(eval_by_method) | set(log_by_method))

        for method in methods:
            if include_methods and method not in include_methods:
                continue
            metrics = eval_by_method.get(method, {})
            costs = log_by_method.get(method, {})
            row = {
                "result_dir": result_dir.name,
                "model": model,
                "dataset": dataset,
                "method": method,
                "method_label": normalize_method_name(method),
                "ndcg_cut_10": metrics.get("ndcg_cut_10"),
                "ndcg_cut_100": metrics.get("ndcg_cut_100"),
                "map_cut_10": metrics.get("map_cut_10"),
                "map_cut_100": metrics.get("map_cut_100"),
                "recall_1000": metrics.get("recall_1000"),
                "avg_comparisons": costs.get("avg_comparisons"),
                "avg_prompt_tokens": costs.get("avg_prompt_tokens"),
                "avg_completion_tokens": costs.get("avg_completion_tokens"),
                "avg_total_tokens": costs.get("avg_total_tokens"),
                "avg_time_per_query": costs.get("avg_time_per_query"),
                "avg_dual_invocations": costs.get("avg_dual_invocations"),
                "avg_single_invocations": costs.get("avg_single_invocations"),
                "avg_order_robust_windows": costs.get("avg_order_robust_windows"),
                "avg_extra_orderings": costs.get("avg_extra_orderings"),
                "avg_regularized_worst_moves": costs.get("avg_regularized_worst_moves"),
            }
            rows.append(row)
    return rows


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def aggregate_by_method(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    aggregates = []
    for method, method_rows in sorted(grouped.items()):
        aggregates.append({
            "method": method,
            "method_label": method_rows[0]["method_label"],
            "n_configs": len(method_rows),
            "ndcg_cut_10": mean(row["ndcg_cut_10"] for row in method_rows),
            "ndcg_cut_100": mean(row["ndcg_cut_100"] for row in method_rows),
            "map_cut_10": mean(row["map_cut_10"] for row in method_rows),
            "map_cut_100": mean(row["map_cut_100"] for row in method_rows),
            "recall_1000": mean(row["recall_1000"] for row in method_rows),
            "avg_comparisons": mean(row["avg_comparisons"] for row in method_rows),
            "avg_prompt_tokens": mean(row["avg_prompt_tokens"] for row in method_rows),
            "avg_completion_tokens": mean(row["avg_completion_tokens"] for row in method_rows),
            "avg_total_tokens": mean(row["avg_total_tokens"] for row in method_rows),
            "avg_time_per_query": mean(row["avg_time_per_query"] for row in method_rows),
            "avg_dual_invocations": mean(row["avg_dual_invocations"] for row in method_rows),
            "avg_order_robust_windows": mean(row["avg_order_robust_windows"] for row in method_rows),
            "avg_regularized_worst_moves": mean(row["avg_regularized_worst_moves"] for row in method_rows),
        })
    return aggregates


def pareto_frontier(rows: List[Dict[str, object]], quality_key: str, cost_key: str) -> List[Dict[str, object]]:
    valid = [
        row for row in rows
        if row.get(quality_key) is not None and row.get(cost_key) is not None
    ]
    frontier = []
    for candidate in valid:
        dominated = False
        for other in valid:
            if other is candidate:
                continue
            better_or_equal_quality = other[quality_key] >= candidate[quality_key]
            lower_or_equal_cost = other[cost_key] <= candidate[cost_key]
            strictly_better = (
                other[quality_key] > candidate[quality_key]
                or other[cost_key] < candidate[cost_key]
            )
            if better_or_equal_quality and lower_or_equal_cost and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: (row[cost_key], -row[quality_key]))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: Optional[float], digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def plot_scatter(rows: List[Dict[str, object]], cost_key: str, output_path: Path) -> None:
    if plt is None or not rows:
        return

    x = [row[cost_key] for row in rows if row.get(cost_key) is not None and row.get("ndcg_cut_10") is not None]
    y = [row["ndcg_cut_10"] for row in rows if row.get(cost_key) is not None and row.get("ndcg_cut_10") is not None]
    labels = [row["method_label"] for row in rows if row.get(cost_key) is not None and row.get("ndcg_cut_10") is not None]
    if not x:
        return

    frontier = pareto_frontier(rows, "ndcg_cut_10", cost_key)
    frontier_points = {
        (row[cost_key], row["ndcg_cut_10"]) for row in frontier
        if row.get(cost_key) is not None and row.get("ndcg_cut_10") is not None
    }

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.65, color="#2c7fb8")
    for x_val, y_val, label in zip(x, y, labels):
        plt.annotate(label, (x_val, y_val), fontsize=8, xytext=(4, 3), textcoords="offset points")

    if frontier_points:
        frontier_x = [point[0] for point in sorted(frontier_points)]
        frontier_y = [point[1] for point in sorted(frontier_points)]
        plt.plot(frontier_x, frontier_y, color="#d95f0e", linewidth=2, marker="o")

    plt.xlabel(FRONTIER_COSTS[cost_key].title())
    plt.ylabel("NDCG@10")
    plt.title(f"Quality-Cost Pareto ({FRONTIER_COSTS[cost_key]})")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_markdown_report(rows: List[Dict[str, object]], method_means: List[Dict[str, object]]) -> str:
    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model"] for row in rows})

    lines = [
        "# Quality-Cost Pareto Analysis",
        "",
        f"- Configurations: {len(rows)}",
        f"- Models: {', '.join(models)}",
        f"- Datasets: {', '.join(datasets)}",
        "",
        "## Mean by Method",
        "",
        "| Method | N Configs | NDCG@10 | Comparisons | Total Tokens | Time(s) | Frontier(comps) | Frontier(tokens) | Frontier(time) |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]

    frontier_membership = {
        cost_key: {row["method"] for row in pareto_frontier(method_means, "ndcg_cut_10", cost_key)}
        for cost_key in FRONTIER_COSTS
    }
    for row in method_means:
        lines.append(
            "| {method_label} | {n_configs} | {ndcg} | {comps} | {tokens} | {time} | {f_comp} | {f_tok} | {f_time} |".format(
                method_label=row["method_label"],
                n_configs=row["n_configs"],
                ndcg=format_metric(row["ndcg_cut_10"]),
                comps=format_metric(row["avg_comparisons"], 1),
                tokens=format_metric(row["avg_total_tokens"], 1),
                time=format_metric(row["avg_time_per_query"], 2),
                f_comp="yes" if row["method"] in frontier_membership["avg_comparisons"] else "",
                f_tok="yes" if row["method"] in frontier_membership["avg_total_tokens"] else "",
                f_time="yes" if row["method"] in frontier_membership["avg_time_per_query"] else "",
            )
        )

    lines.extend([
        "",
        "## Frontier Members",
        "",
    ])
    for cost_key, label in FRONTIER_COSTS.items():
        frontier = pareto_frontier(method_means, "ndcg_cut_10", cost_key)
        lines.append(f"### By {label.title()}")
        lines.append("")
        for row in frontier:
            lines.append(
                f"- `{row['method_label']}`: NDCG@10={format_metric(row['ndcg_cut_10'])}, "
                f"{label}={format_metric(row[cost_key], 2)}"
            )
        lines.append("")

    per_config_frontiers = defaultdict(dict)
    for result_dir in sorted({row["result_dir"] for row in rows}):
        config_rows = [row for row in rows if row["result_dir"] == result_dir]
        for cost_key in FRONTIER_COSTS:
            per_config_frontiers[result_dir][cost_key] = pareto_frontier(config_rows, "ndcg_cut_10", cost_key)

    lines.append("## Per-Configuration Frontier Members")
    lines.append("")
    for result_dir in sorted(per_config_frontiers):
        lines.append(f"### {result_dir}")
        lines.append("")
        for cost_key, label in FRONTIER_COSTS.items():
            members = ", ".join(f"`{row['method_label']}`" for row in per_config_frontiers[result_dir][cost_key])
            lines.append(f"- {label.title()}: {members or 'none'}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result_dirs = args.result_dirs or discover_result_dirs(args.results_root)
    include_methods = set(args.include_methods) if args.include_methods else None

    rows = collect_points(result_dirs, include_methods=include_methods)
    if not rows:
        raise SystemExit("No result points found. Check --results_root / --result_dirs.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    method_means = aggregate_by_method(rows)
    write_csv(args.output_dir / "all_points.csv", rows)
    write_csv(args.output_dir / "method_means.csv", method_means)

    for cost_key in FRONTIER_COSTS:
        frontier_rows = pareto_frontier(method_means, "ndcg_cut_10", cost_key)
        write_csv(args.output_dir / f"frontier_{cost_key}.csv", frontier_rows)
        plot_scatter(method_means, cost_key, args.output_dir / f"pareto_{cost_key}.png")

    report = build_markdown_report(rows, method_means)
    (args.output_dir / "QUALITY_COST_PARETO.md").write_text(report)
    print(f"Wrote Pareto analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
