# Field Gap Map

<!-- AUTO-GENERATED VIEW. The `Addressed by` column reflects the current state of
`addresses_gap` edges in `graph/edges.jsonl`. Do not hand-edit; regenerate after
mutating the graph. Claims are evidence, not addressers — they do not appear here. -->

Stable-ID catalog of field gaps this project targets. Only papers and ideas address a gap (via the `addresses_gap` edge type). Claim status about a gap is captured on the claim pages themselves.

| Gap ID | Statement                                                                                                                                                          | Status                        | Addressed by (from graph)                                                                                             |
|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| gap:G1 | Standard setwise LLM ranking extracts only one decision per call (identity of best); implicit information about worst and relative order of the rest is discarded. | partial                       | paper:zhuang2024_setwise, idea:001, idea:002, idea:003, idea:004, idea:006, idea:007, idea:008                        |
| gap:G2 | Position bias in setwise ranking differs for best vs worst selection; no prior study of bias under joint best+worst prompting.                                     | partial                       | paper:tang2024_found_in_middle, idea:002, idea:005                                                                    |
| gap:G3 | No systematic study of whether LLMs are symmetrically competent at identifying most vs least relevant documents.                                                   | diagnosed; mitigation pending | idea:001 (diagnosis-via-failure), idea:002 (partial mitigation), idea:006                                             |
| gap:G4 | No framework for when expensive setwise variants are justified over baseline heapsort/bubblesort.                                                                  | partial (frontier mapped)     | paper:podolak2025_setwise_insertion, paper:peng2025_flops_reranking, idea:004, idea:005, idea:006, idea:007, idea:008 |
| gap:G5 | LLM ranking performance may vary by model family, but prior work treats all models uniformly.                                                                      | partial (pattern identified)  | paper:hutter2025_positional_rag, idea:004, idea:005, idea:006, idea:008                                               |

## Detailed gap notes

### gap:G1 — Information extraction per setwise call

**Motivating observation.** Setwise prompts elicit internal reasoning about all candidates; the method uses only the "best" label. At n=4 candidates, a full ranking contains log₂(4!) ≈ 4.58 bits; a single-best label gives log₂(4) = 2 bits.

**Strategies tried in this project:**

- Reverse-elicit worst (idea:001) → failed; worst alone is biased and unreliable.
- Co-elicit best+worst in one prompt (idea:002) → partial success; DualEnd wins 14/18 configs but statistically fragile (see claim:C6).
- Fuse two independent TD/BU rankings (idea:003) → failed; imports BU bias.

**Status:** Partial. Joint elicitation (idea:002, Qwen-generation path only) extracts more information per call but is expensive (5–9× wall-clock). Selective / same-call refinements (idea:004, idea:006) aim to close the efficiency gap. MaxContext (idea:007) is the active whole-pool variant for extracting more information per call by fitting up to 50 Qwen passages into a single prompt. See claim:C2, claim:C8 for the positive story and claim:C6 for the significance caveat.

### gap:G2 — Positional bias under joint prompts

Prior position-bias work (paper:liu2024_lost_in_middle, paper:tang2024_found_in_middle, paper:hutter2025_positional_rag) covers single-objective prompts only. The `dual_worst` primacy reversal (documented by claim:C5 and exp:analysis_position_bias) is a novel observation.

**Status:** Diagnosed; mitigation (idea:005 bias_aware_dualend, all runs pending) not yet validated.

### gap:G3 — Asymmetry of best-vs-worst competence

exp:main_bu_heap and exp:main_bu_bubble (along with the position-bias analysis) provide the first systematic evidence — captured in claim:C1, claim:C3 — that LLMs are **not** symmetrically competent: standalone worst-selection collapses to position heuristics. Joint co-elicitation (idea:002) rescues the worst signal partially, and same-call regularization (idea:006, pending) proposes using the worst signal as a local demotion only.

**Status:** Diagnosed strongly; mitigation path (co-elicitation + same-call regularization) pending validation.

### gap:G4 — Setwise efficiency-effectiveness frontier

paper:podolak2025_setwise_insertion and paper:peng2025_flops_reranking frame setwise efficiency but do not address the "more info per call" axis. The Pareto analysis (exp:analysis_pareto, backed by claim:C9) identifies an empty region between TD-Bubble and DE-Cocktail that refinement methods (idea:004/005/006) and MaxContext (idea:007) target.

**Status:** Frontier mapped; refinements and MaxContext matrix pending.

### gap:G5 — Model-family-dependent performance

paper:hutter2025_positional_rag predicts that bias profiles differ by model family. exp:ablation_num_child and the main sweeps show T5 and Qwen respond differently to window size (captured in claim:C7) and to DualEnd (captured in claim:C2). The selective / bias-aware / same-call refinements (idea:004/005/006) implicitly route by model through query-local gates; a global model-family router does not yet exist.

**Status:** Pattern identified; global routing mechanisms pending.
