---
type: claim
node_id: claim:C7
statement: "Optimal setwise window size (num_child) interacts with model family: T5 models (512-token context) prefer small windows (nc=2–3); Qwen models (32k+ context) are stable across sizes or prefer slightly larger windows."
status: supported
evidence_strength: medium
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Claim

DualEnd-Cocktail DL19 by num_child:

| Model | nc=2 | nc=3 | nc=5 | nc=7 |
|---|---:|---:|---:|---:|
| flan-t5-xl | 0.6988 | 0.6884 | 0.6749 | 0.6480 |
| qwen3-8b | 0.7187 | 0.7155 | 0.7224 | 0.7249 |
| qwen3.5-9b | 0.7392 | 0.7370 | 0.7336 | 0.7386 |

- T5 shows monotone degradation with larger windows (context truncation).
- Qwen is flat-to-slightly-up with larger windows.

## Supporting experiments

- exp:ablation_num_child

## Caveats

- Tested on 3 models only (one per context-size regime); pattern is suggestive, not universal.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
