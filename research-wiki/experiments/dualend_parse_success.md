---
type: experiment
node_id: exp:dualend_parse_success
title: "Analysis — DualEnd joint-prompt parse reliability (Qwen generation path)"
status: completed
config:
  metric: "rate of \"partial dual parse\" and \"only parse one\" warnings emitted by `_parse_dual_output` over Qwen generation runs; see `llmrankers/setwise_extended.py:638+`"
  models_measured: ["flan-t5 (all sizes)", "qwen3 (4b/8b/14b)", "qwen3.5 (4b/9b/27b)"]
  source_artifacts: ["research_pipeline_setwise/FINDINGS.md (lines 164-171)"]
tests: ["idea:002", "claim:C2", "claim:C8"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:40:00+10:00
updated_at: 2026-04-20T10:46:00+10:00
---

# Summary

Diagnostic over the Qwen generation path (the only code path that actually elicits joint best+worst). Counts calls where the model emitted something other than the canonical `Best: X, Worst: Y` string, measured by the cascading fallbacks inside `_parse_dual_output` in `llmrankers/setwise_extended.py:566-638+`.

## Result (from FINDINGS.md:164-171)

- **All Flan-T5 models:** 0 parse failures, 0 unexpected outputs across all datasets. (Flan-T5 DualEnd does not decode the dual string at all — it uses the best-only-proxy shortcut from `setwise_extended.py:445-453`, so there is nothing to parse.)
- **Qwen3-4B:** 0–2 "only parse one" warnings per ~33K comparisons. Near-perfect.
- **Qwen3-14B DL20:** the noisiest case — 254 "only parse one" + 218 "partial dual parse" for cocktail across ~29K comparisons (~0.9%). DL19 much cleaner (57 + 56).
- **Qwen3.5-27B:** very clean, 13 "only parse one" across ~23K comparisons (DL19).
- **No model has actual "dual parse failures":** the cascading parser always produces some best/worst pair.

## Implication

Parsing reliability is not a confound for the paper's claims. The <1% partial-parse rate on the worst configuration is well within noise.

The separate confound worth disclosing is **not** parse failure but **code-path divergence**: T5 and all `--scoring likelihood` paths silently route through the best-only-proxy shortcut (`:476-491`). Only Qwen `--scoring generation` is a true joint elicitation. This is captured on `idea:002` and in claim:C10 framing constraints.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
