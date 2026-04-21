---
type: claim
node_id: claim:C8
statement: "The core contribution of this project is joint elicitation (asking for best and worst in the same prompt), not sorting-algorithm novelty — cocktail-shaker and double-ended selection are implementation details of the elicitation idea."
status: supported
evidence_strength: high
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Claim

Three facts jointly identify joint elicitation as the load-bearing contribution:

1. **BottomUp alone fails** — worst-selection is unreliable when elicited independently (claim:C1, claim:C3).
2. **BiDir fails** — fusing TD and BU independently cannot rescue the worst signal (claim:C4).
3. **DualEnd partially succeeds** — worst signal is useful only when elicited in the same prompt as best (claim:C2).

The cocktail-shaker sort and double-ended selection sort are the *consumers* of the dual output; they are necessary plumbing but not the scientific contribution. The contribution is the observation that worst information must be *co-elicited* with best.

## Paper-framing implication

- Lead with the elicitation result, not the sort.
- Frame BottomUp and BiDir as coherent negative results that isolate the mechanism.
- Position `DE-Cocktail` as a quality-first option; leave efficiency to selective variants (idea:004).

## Supporting claims

- claim:C1, claim:C2, claim:C3, claim:C4, claim:C5.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
