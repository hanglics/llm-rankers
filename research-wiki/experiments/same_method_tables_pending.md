---
type: experiment
node_id: exp:same_method_tables_pending
title: "Same-method/same-sort consolidated result tables (completed 2026-04-21)"
status: completed
config:
  description: "Per Need_to_Run.txt top priority: build result tables that fix a single (method × sort × model × dataset) cell rather than reporting only best-per-family."
  rationale: "Paper and UI now need like-with-like comparisons (e.g. BU-Bubble vs DE-Bubble vs TD-Bubble at matched sort)."
  source_artifacts:
    - "Need_to_Run.txt (lines 3-4)"
    - "research_pipeline_setwise/SIGNIFICANCE_TESTS_PAIRWISE.md"
    - "research_pipeline_setwise/SIGNIFICANCE_TESTS_PAIRWISE.json"
    - "analysis/significance_tests_pairwise.py"
    - "results-display/index.html (section id=\"pairwise-tables\")"
tests: ["claim:C2", "claim:C6"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:40:00+10:00
updated_at: 2026-04-21T19:50:00+10:00
---

# Summary

**Completed on 2026-04-21.** Produced 12 pairwise comparison tables (6 groupings × 2 datasets) reporting NDCG@10 + Bonferroni-adjusted significance for specific (baseline, challenger) pairs at matched sorting algorithm. Each table is its own Bonferroni family (per grouping × per dataset).

Groupings:

1. **TD-Bubble baseline** vs BU-Bubble, DE-Cocktail (DL19 + DL20)
2. **TD-Heap baseline** vs BU-Heap (DL19 + DL20)
3. **TD-Bubble baseline** vs DE-Cocktail, DE-Selection (DL19 + DL20)
4. **TD-Heap baseline** vs DE-Cocktail, DE-Selection (DL19 + DL20)
5. **TD-Bubble baseline** vs BiDir-RRF, BiDir-Wt(a=0.7) (DL19 + DL20)
6. **TD-Heap baseline** vs BiDir-RRF, BiDir-Wt(a=0.7) (DL19 + DL20)

## Headline findings

Cleanest positive result: **DualEnd vs TD-Bubble on DL19** — mean Δ +0.0043, 10/18 positive, **2 Bonferroni-sig wins** (Qwen3-8B DE-Cocktail; Qwen3-8B DE-Selection). All other DualEnd groupings are directionally positive but mostly +ns. BU-Bubble, BU-Heap, and both BiDir arms all show multiple Bonferroni-sig losses — consistent with existing family-best findings.

## Method

- Statistical test: paired approximate randomization (100K samples).
- 95% bootstrap CI (20K resamples).
- Bonferroni correction **inside each (grouping, dataset) table** over 9 models × |challengers| tests.
- Reuses `analysis/significance_tests.py` helpers; new driver at `analysis/significance_tests_pairwise.py`.

## Artifacts

- **MD:** `research_pipeline_setwise/SIGNIFICANCE_TESTS_PAIRWISE.md` — per-model rows with delta, CI, p-values, verdicts.
- **JSON:** `research_pipeline_setwise/SIGNIFICANCE_TESTS_PAIRWISE.json` — keyed by `{grouping_id}__{dataset}`.
- **HTML fragment:** `results-display/_pairwise_tables_fragment.html` — regenerable; inlined into `results-display/index.html` under section `id="pairwise-tables"`.

## How to regenerate

```bash
./ranker_env/bin/python analysis/significance_tests_pairwise.py
# Then re-run the one-off inline insert if index.html needs refresh.
```

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
