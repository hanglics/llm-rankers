# Research Log — EMNLP 2026 Short Paper

Active log scoped to the EMNLP work. Legacy log entries (IDEA_007, Phase 4,
ablations) live at `Extra-Experiments/research-wiki/log.md`.

## 2026-05-07
- **Project restructure**: isolated EMNLP active scope. Moved legacy
  experiments / analysis / results / wiki content / planning docs to
  `Extra-Experiments/`. Renamed `EMNLP_PAPER_DESIGN.md` →
  `EMNLP_PAPER_DESIGN.md`, `EMNLP_IMPLEMENTATION_PLAN.md` →
  `EMNLP_IMPLEMENTATION_PLAN.md`, `paper/v2/` → `paper/v1/`. See
  `pre-emnlp-restructure` git tag for rollback point.
- **Phase F position-bias controls** (--shuffle, --reverse) implemented
  for MaxContext methods only; Heap/Bubble untouched. Fixed seed 929
  via `hashlib.blake2b`; per-comparison ordering with label remapping
  to original window indices.

## 2026-05-06
- **Multimodal loader refactor (Phase 3a)** for Mistral 3 + Qwen 3.5
  vision-language configs (`AutoProcessor` + `AutoModelForImageTextToText`).
  `qwen3` / `qwen3_moe` remain on causal path (byte-equal preserved).
- Logging filter for `max_new_tokens vs max_length` warning at
  `setwise.py` module top (transformers emits via `logger.warning`,
  not `warnings.warn`).
- Soft-refusal recovery in `_parse_single_label` for verbose chat
  models (Llama-3.1, Mistral-3): if output has refusal phrase AND
  choice indicator, extract trailing numeric answer instead of
  defaulting to passage 1.

## Earlier (legacy archive)
See `<repo-root>/Extra-Experiments/research-wiki/log.md` for the IDEA_007
and pre-EMNLP log history.
