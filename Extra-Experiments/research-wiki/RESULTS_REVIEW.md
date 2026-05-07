# Results Review V1

## Scope

This review is based on the current project materials in [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md), [NARRATIVE.md](NARRATIVE.md), [FINDINGS.md](FINDINGS.md), [PAPER_PLAN.md](PAPER_PLAN.md), and [SIGNIFICANCE_TESTS.md](SIGNIFICANCE_TESTS.md), together with the saved outputs under `results/`.

BEIR experiments are still running, so this review focuses on the completed TREC DL results and what they currently support.

## Executive Verdict

This is a real paper, but it is not yet strongest as a pure "new method clearly beats the baseline" paper.

The current evidence supports a more careful and more interesting story:

- `DualEnd` is the strongest family overall.
- `BottomUp` is systematically unreliable.
- `BiDir` fails because it imports BottomUp noise.
- The most interesting scientific observation is not just that DualEnd can help, but that **worst-selection behaves very differently when asked alone versus when asked jointly with best-selection**.

Right now, the work is strongest as an **analysis-driven IR paper with one modestly effective method and two coherent negative results**. That is a better fit for `ICTIR` than for a more ambitious later `ARR` submission.

## Venue Fit

### ICTIR

This looks plausible for ICTIR if the paper is framed around:

- directional asymmetry in setwise LLM ranking,
- the failure mode of standalone worst-selection,
- the limited but repeatable strength of joint best-and-worst elicitation,
- and the resulting quality-cost tradeoff.

ICTIR is a good home for a careful study that combines method design, diagnostic analysis, and negative results, especially if the narrative is honest and the conclusions are scoped conservatively.

### Later ARR

The current package is not yet strong enough for ARR if the main pitch is "DualEnd is a better ranking algorithm." The main obstacles are:

- only one Bonferroni-significant gain,
- substantial efficiency cost,
- limited current generalization evidence outside TREC DL,
- and a contribution that still reads partly as a sorting tweak rather than a broader insight about LLM ranking behavior.

For ARR, the work likely needs either:

- a stronger and more efficient refined method, or
- a broader conceptual contribution that generalizes beyond this single reranking setup.

## What Is Strong

### 1. The experimental discipline is strong

The project is more mature than a typical early-stage method paper:

- 9 models across 3 families,
- 2 TREC DL datasets,
- multiple method families,
- multiple ablations,
- position bias analysis,
- ranking agreement analysis,
- per-query win analysis,
- and explicit significance testing.

This gives the work credibility even where the positive gains are modest.

### 2. The negative results are coherent

The failure modes are not random or noisy in a confusing way. They form a consistent picture:

- BottomUp is weaker than TopDown in all 18 configs.
- BottomUp has 6 Bonferroni-significant losses.
- BiDir is usually worse because it relies on BottomUp.
- DualEnd is much closer to TopDown than BottomUp is.

That coherence is valuable. It means the paper can tell a clean scientific story rather than reporting a bag of mixed outcomes.

### 3. The most interesting finding is the asymmetry of worst-selection

This is the most novel part of the project.

Worst-selection alone is weak and heavily recency-biased, but worst-selection inside a joint DualEnd prompt behaves differently and even flips the bias pattern. That is more interesting than "cocktail sort works well."

If emphasized properly, this becomes a paper about **how LLMs perform comparative ranking judgments**, not just about using a different sorting routine.

### 4. The Qwen-side pattern is meaningful

Even though only one win survives Bonferroni correction, the fact that DualEnd wins all `12/12` Qwen model-dataset configurations is still a meaningful directional pattern. Reviewers may not accept it as strong proof of universal superiority, but it is absolutely enough to motivate a more refined method or a broader hypothesis.

## What Is Weak

### 1. The main positive claim is statistically fragile

This is the biggest issue.

The current evidence says:

- DualEnd is positive in `14/18` configurations.
- Mean delta versus best TopDown is `+0.0058`.
- Only `qwen3-4b` DL19 survives Bonferroni correction.

That means the paper should not claim strong, general, statistically established superiority on TREC DL.

### 2. The efficiency story cannot carry the paper

DualEnd is not efficient in the practical end-to-end sense:

- `DE-Cocktail` is about `8.89x` slower than `TD-Heap` on average.
- `DE-Selection` is about `5.60x` slower than `TD-Heap` on average.

The "more information per call" argument is intellectually interesting, but by itself it is not enough when the actual budgeted runtime and comparison count are much worse.

### 3. The paper is currently spread across too many equally weighted ideas

The current framing still gives BottomUp, DualEnd, and BiDir similar conceptual weight. The results no longer support that.

In reality:

- DualEnd is the only strategy that should remain central.
- BottomUp should be presented as a diagnostic negative result.
- BiDir should be presented as a downstream failure caused by BottomUp noise.

The paper becomes stronger when it focuses on one core idea rather than three parallel "contributions."

### 4. Sorting novelty alone is not enough

The algorithmic novelty is real, but likely not enough on its own. "We adapted cocktail shaker sort and double-ended selection sort to dual-output LLM comparisons" is a reasonable systems detail, but not the core scientific contribution reviewers will remember.

The stronger contribution is the behavioral finding about **joint elicitation**.

## What Claims Are Safe

The following claims are currently well supported:

- DualEnd is the strongest family overall on the completed TREC DL experiments.
- BottomUp is not a viable standalone replacement for TopDown.
- BiDir does not help because BottomUp is too noisy.
- Joint elicitation changes positional bias and likely changes the model's comparison behavior.
- DualEnd should be treated as a quality-first option, not an efficiency-first one.

The following claims are currently too strong:

- DualEnd is a generally statistically significant improvement.
- DualEnd is practically efficient.
- Extracting more information per call reliably translates to stronger end-to-end ranking.
- The method is broadly general without further benchmark evidence.

## Novelty Assessment

### What feels novel

- Studying best-selection, worst-selection, and joint best-worst selection within one setwise ranking framework.
- Showing that worst-selection alone is poor but can become somewhat useful when elicited jointly.
- The dual-worst bias reversal.
- The empirical separation between "same-call dual signal" and "independent fusion signal."

### What does not feel sufficiently novel on its own

- Simply introducing cocktail sort or dual-ended selection into the setwise pipeline.
- Simply observing modest average gains.
- Simply arguing from information-theoretic bits per call without budget-aware evidence.

### The strongest novelty framing

The strongest framing is:

> LLM setwise ranking is directionally asymmetric. Worst-selection is not a reliable standalone signal, but joint elicitation of best and worst changes the model's behavior and can modestly improve ranking while altering positional bias.

That is much stronger than:

> We tried a new sorting algorithm and it sometimes helps.

## Interpretation of the Current Results

The data suggests the useful part of DualEnd is not that worst-selection becomes independently accurate. Instead, the likely mechanism is:

- the model still relies primarily on best-selection,
- the joint prompt changes the comparison regime,
- and the additional worst output provides a weak auxiliary constraint or disambiguation signal.

This interpretation is supported by:

- higher agreement between TopDown and DualEnd than between TopDown and BottomUp,
- the failure of standalone BottomUp,
- the failure of independent BiDir fusion,
- and the changed positional bias under dual prompting.

That mechanism story is promising, but it is not yet fully nailed down. Strengthening it would materially improve the paper.

## Highest-Leverage Improvements

### 1. Build a selective DualEnd method

This is the best next move.

Use TopDown by default and invoke DualEnd only when it is most likely to matter:

- on uncertain windows,
- near the top-k decision boundary,
- on the last reranking stage over a small shortlist,
- or when score margins between candidate documents are small.

Why this matters:

- It directly addresses the biggest weakness, which is cost.
- It is much easier to sell than full DualEnd everywhere.
- It converts the current result into a quality-cost tradeoff method rather than a brute-force expensive variant.

This is probably the highest acceptance-lift-per-effort direction.

### 2. Turn the position-bias finding into a method

Right now the bias analysis is interesting but mostly descriptive. A stronger move is to exploit it.

Possible direction:

- run DualEnd with light position perturbation only on uncertain windows,
- aggregate joint best/worst outputs across a few orderings,
- and use this as a bias-robust reranking component.

That would convert the paper from "we observed a novel bias pattern" to "we observed it and used it."

### 3. Use same-call worst information, not standalone BottomUp

The current results strongly suggest that the worst signal is only useful when tightly coupled with the best signal in the same prompt.

So the next refinement should not be:

- more BottomUp work, or
- more independent TopDown/BottomUp fusion.

It should instead be:

- better extraction and use of the worst signal inside the same DualEnd comparison,
- possibly as a regularizer, confidence signal, or local pruning rule.

### 4. Report a quality-cost Pareto frontier

This is essential.

The current tables make it too easy for a reviewer to say:

> This is a tiny gain for huge cost.

You should instead show:

- quality versus comparisons,
- quality versus completion tokens,
- quality versus wall time,
- and quality versus number of DualEnd invocations if you implement a selective variant.

This will let you argue much more precisely about whether the method is worthwhile under different budgets.

### 5. Add targeted mechanism analysis

You already know DualEnd is not uniformly better. The next question is when and why it helps.

A strong analysis would categorize windows or queries where DualEnd beats TopDown:

- one obvious distractor plus several plausible candidates,
- near-duplicate passages,
- factoid versus compositional queries,
- narrow versus broad intents,
- easy, medium, and hard queries with stronger qualitative inspection,
- windows with high versus low TopDown confidence.

If you can show that DualEnd helps in specific identifiable regimes, the paper becomes much more convincing even if the average gain stays modest.

## Lower-Priority Directions

The following are currently low-value:

- spending more effort on standalone BottomUp,
- more alpha tuning for BiDir,
- more presentation polish around "bits per call" without improving budgeted performance,
- and treating all three strategies as equally important contributions.

These directions are unlikely to change the core acceptance picture.

## Suggested Reframe for the Paper

The paper should be re-centered around the following question:

> What happens when we ask LLMs for both the best and the worst item in setwise ranking, rather than only the best?

Then the paper can answer:

1. Standalone worst-selection is weak and biased.
2. Joint best-worst elicitation changes the behavior of the model.
3. That joint elicitation can modestly improve ranking quality.
4. The gains are real but expensive.
5. The main opportunity is not standalone BottomUp or late fusion, but selective or bias-aware use of the joint signal.

This framing is cleaner, more honest, and more publishable than presenting three parallel bidirectional strategies.

## Immediate Next Experiments

If the goal is to maximize paper strength with limited extra effort, I would prioritize the following:

1. Implement `Selective DualEnd`.
   Run DualEnd only on a shortlist or only on uncertain windows. Measure quality-cost curves.

2. Implement a light `Order-Robust DualEnd`.
   Use a small number of controlled permutations on uncertain windows only. Test whether this improves robustness enough to justify the added cost.

3. Perform a `When DualEnd Helps` analysis.
   Inspect wins and losses at the query or window level and group them into a small taxonomy with examples.

4. Finish `BEIR` and report the results carefully.
   If the directional pattern persists across more datasets, the current statistical fragility on TREC DL becomes much less damaging.

5. Rebuild the narrative around `joint elicitation`, not around sorting novelty.

## Submission Strategy

### For ICTIR

Submit if:

- BEIR is at least directionally supportive,
- the paper is reframed around directional asymmetry and joint elicitation,
- and ideally one selective DualEnd refinement is added.

Without any extra method refinement, the paper can still be viable for ICTIR if the analysis sections are strong and the claims remain conservative.

### For Later ARR

Wait unless at least one of the following happens:

- a refined DualEnd variant materially improves the quality-cost tradeoff,
- BEIR shows stronger and broader gains,
- or the conceptual framing is expanded beyond this one reranking benchmark family.

ARR will likely demand either a stronger method or a more general theory/phenomenon story.

## Bottom Line

This work is promising, but its best version is not yet "we found a new ranking algorithm that clearly wins."

Its best current version is:

- a strong analysis of directional asymmetry in setwise LLM ranking,
- a modest but repeatable joint-elicitation improvement,
- a clear explanation of why standalone worst-selection fails,
- and a path toward a better refined method based on selective or bias-aware DualEnd usage.

For `ICTIR`, that can be enough.

For a later `ARR`, I would push for one more refinement cycle focused on:

- selective DualEnd,
- bias-aware or order-robust DualEnd,
- and stronger evidence about when the joint signal helps.
