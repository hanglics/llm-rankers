---
type: claim
node_id: claim:C10
statement: "The project targets ICTIR 2026 with a conservative analysis-driven framing (one modestly effective method + two coherent negative results); ARR submission is gated on a stronger refinement / generalization package."
status: reported
evidence_strength: policy
origin_skill: manual-backfill
created_at: 2026-04-20T10:40:00+10:00
updated_at: 2026-04-30T20:10:00+10:00
---

# Claim (framing-level, not empirical)

This is a framing / policy claim, not an experimental hypothesis. It captures what the paper is allowed to say given the evidence on disk.

## Target venue and story shape

- **Primary target:** ICTIR 2026 (per [`RESEARCH_BRIEF.md`](../RESEARCH_BRIEF.md), [`PAPER_PLAN.md`](../PAPER_PLAN.md), [`RESULTS_REVIEW.md`](../RESULTS_REVIEW.md)).
- **Conditional target:** ARR only if BEIR generalization (exp:beir_generalization) and refinement methods (idea:004/005/006) land positively.
- **Story shape:** analysis-driven IR paper with one modestly effective method (idea:002 DualEnd) and two coherent negative results (idea:001 BottomUp, idea:003 BiDir). Not a "new state-of-the-art" paper.

## Hard framing constraints implied by the evidence

- **Do not** claim DualEnd is universally better. It is a directional pattern (14/18 configs) with only one Bonferroni-significant win — see claim:C6.
- **Do not** claim DualEnd is more efficient. It is 5–9× slower than TD-Heap. Framing must lead with quality; efficiency is the trade-off.
- **Do not** claim the worst-selection inside DualEnd is independent of the best-selection. On all paths except Qwen-generation, the `--scoring likelihood` / T5 code falls back to a best-only proxy — see idea:002 mechanism notes.
- **Do** lead with directional asymmetry (claim:C1), joint elicitation as the core contribution (claim:C8), and the novel dual_worst primacy reversal (claim:C5).
- **Do** present BU and BiDir as evidence that isolates the mechanism, not as "methods that didn't work" prose.

## Open decision points (from `NARRATIVE.md` and `RESULTS_REVIEW.md`)

1. BEIR generalization landing — if DualEnd pattern replicates out-of-domain, the paper strengthens materially (exp:beir_generalization).
2. Selective DualEnd quality-cost tradeoff — if idea:004 hits the empty region on claim:C9's frontier, it becomes the primary method.
3. Bias-aware DualEnd evidence — if idea:005 exploits the dual_worst primacy reversal, that's a mechanistic contribution story.
4. MaxContext family evidence — if idea:007 matches or beats same-depth baselines at lower comparison / wall-clock cost, it can become the main forward method while the original DualEnd / BottomUp / BiDir work becomes supporting evidence and ablation.

## Closed decision points

- Same-method / same-sort result tables (former `Need_to_Run.txt` top priority) were completed on 2026-04-21 as exp:same_method_tables_pending. They reinforce the conservative framing: the cleanest positive result is DualEnd vs TD-Bubble on DL19 with 2 Bonferroni-significant Qwen3-8B wins, while BottomUp and BiDir same-sort groupings show multiple significant losses.

## Why this is a claim and not just a note

Framing decisions degrade silently. Capturing them as a `claim:` with `status: reported` means any round-2 auditor or downstream pipeline (`/paper-plan`, `/idea-creator`) sees the constraints explicitly and does not re-introduce overclaims.

## Supporting experiments

- exp:analysis_significance_tests (defines the statistical ceiling)
- exp:beir_generalization (gates the story upgrade)

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
