---
type: experiment
node_id: exp:likelihood_scoring_pending
title: "Likelihood-scoring follow-ups (flan-t5-xl, Qwen3-8B, Qwen3.5-9B, DL19 & DL20)"
status: not_submitted
config:
  models: ["flan-t5-xl", "Qwen3-8B", "Qwen3.5-9B"]
  datasets: ["dl19", "dl20"]
  scoring: likelihood
tests: ["idea:002"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Status

6 runs pending (3 models × 2 datasets).

**What each path actually exercises** (see `setwise_extended.py:476-491` and `README.md:423-427`):

- `flan-t5-xl --scoring likelihood`: T5 topdown/bottomup use the native Flan-T5 likelihood comparator over short labels (A/B/C/D). This is the paper:zhuang2024_setwise-native path. For DualEnd, it collapses to a best-only proxy via argmin — same proxy as T5 generation.
- `Qwen3-8B`, `Qwen3.5-9B --scoring likelihood`: for DualEnd (and selective / bias-aware / same-call variants), the code reuses `_build_best_prompt` + argmin shortcut. **The worst signal under likelihood is not an independently-scored worst elicitation.**
- **Only Qwen `--scoring generation`** runs the true joint best+worst prompt (idea:002 mechanism notes).

This page tests the likelihood path across all three models, keeping the T5 vs causal distinction explicit to avoid future confusion.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
