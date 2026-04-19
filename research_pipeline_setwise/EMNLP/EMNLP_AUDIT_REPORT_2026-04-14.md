# EMNLP / ARR Audit Report

Date: 2026-04-14  
Workspace: `/Users/hangli/projects/llm-rankers/research_pipeline_setwise/EMNLP/latex`

## Scope

This audit reviews the current EMNLP draft against:

- the current project source of truth in:
  - `/Users/hangli/projects/llm-rankers/research_pipeline_setwise/RESEARCH_BRIEF.md`
  - `/Users/hangli/projects/llm-rankers/research_pipeline_setwise/PAPER_PLAN.md`
  - `/Users/hangli/projects/llm-rankers/research_pipeline_setwise/RESULTS_REVIEW_V1.md`
  - `/Users/hangli/projects/llm-rankers/research_pipeline_setwise/FINDINGS.md`
  - `/Users/hangli/projects/llm-rankers/EXPERIMENT_PLAN.md`
- the current ARR CFP and ACL formatting guidance:
  - ARR CFP: <https://aclrollingreview.org/cfp>
  - ACLPUB formatting guidelines: <https://acl-org.github.io/ACLPUB/formatting.html>
- the current LaTeX source and compiled PDF/log:
  - `/Users/hangli/projects/llm-rankers/research_pipeline_setwise/EMNLP/latex/main.tex`
  - `/Users/hangli/projects/llm-rankers/research_pipeline_setwise/EMNLP/latex/main.log`
  - `/Users/hangli/projects/llm-rankers/research_pipeline_setwise/EMNLP/latex/main.pdf`

The paper was recompiled with `latexmk -pdf -interaction=nonstopmode main.tex` during this audit.

## Executive Verdict

The current EMNLP draft is **not submission-ready for ARR / EMNLP Long Papers**.

The main reasons are:

1. **Policy / compliance blockers**
   - The draft is far over the ARR long-paper limit as written.
   - The abstract exceeds the ACL abstract-length guideline.
   - The paper explicitly includes **in-progress** BEIR work and an **implemented but unevaluated refinement package** as part of the story, which conflicts with ARR’s requirement that long papers describe **completed** work.
   - If the ICTIR version is under review, accepted, or published, the current EMNLP/ARR version is likely blocked by ARR’s multiple-submission / overlap policy unless it becomes a materially different paper.

2. **Narrative / evidence mismatch**
   - The strongest current story is still the one already identified in the project docs: an **analysis-driven IR paper** centered on directional asymmetry, one modest positive method family, and two coherent negative results.
   - The draft still gives too much weight to three equally “novel methods” plus a future refinement package, even though the repo’s own planning docs say the paper is strongest as an asymmetry / mechanism paper.

3. **Real format issues**
   - There are reproducible overfull boxes and one oversized float in the compiled PDF.
   - The “line numbers in the middle of the page” are **not a bug**. They are the expected ACL review-mode gutter ruler. Some content, however, is sitting too close to that ruler because the draft contains wide elements.

## What Is Already Strong

- The paper has a real, coherent core result:
  - `DualEnd` is the strongest family overall on the completed TREC DL study.
  - `BottomUp` is a consistent negative result.
  - `BiDir` fails for a coherent reason.
  - the most interesting scientific observation is the difference between standalone worst-selection and joint best-worst elicitation.
- The analysis sections are the strongest part of the draft.
- The repo already contains the right conservative framing:
  - `RESEARCH_BRIEF.md:33-34` says the target is `ICTIR` first and only “later ARR” after stronger refinement / generalization.
  - `PAPER_PLAN.md:7-8` says the same.
  - `RESULTS_REVIEW_V1.md:20` says the work is currently strongest as an analysis-driven IR paper and a better fit for ICTIR than a more ambitious ARR submission.

## P0: Submission-Blocking Issues

### 1. The current draft is far over ARR long-paper limits

Official ARR CFP:

- long papers must be “substantial, original, completed and unpublished work”
- up to **8 pages of content**
- unlimited extra space only for **Limitations** and optional **Ethical considerations**
- plus unlimited references

Relevant official source:

- ARR CFP `Long Papers`: <https://aclrollingreview.org/cfp>

Current draft status:

- `main.pdf` is **21 pages**
- the main paper content runs to roughly **18 pages before References**, with the `Limitations` section beginning on page 19
- this is not a mild overflow; it is a full draft-length mismatch with ARR long-paper format

Implication:

- “No need to strictly apply page limit for now” is fine for internal drafting.
- But this is not a near-submission draft in ARR terms. It is an exploratory long-form manuscript that still needs a serious reduction pass.

### 2. The abstract is too long for ACL formatting guidance

Official ACLPUB formatting guidance says:

- abstract should be no longer than **200 words**

Relevant official source:

- ACLPUB formatting: <https://acl-org.github.io/ACLPUB/formatting.html>

Current draft status:

- abstract in `main.tex:85-87` is approximately **247 words**

Implication:

- this is a concrete format violation, not just a style preference

### 3. ARR requires completed work; the current draft openly presents in-progress generalization

Official ARR CFP says long papers must describe **completed** work.

Current draft status:

- `experiment_setup.tex:16` says BEIR is “in progress” and “will be folded into the camera-ready version”
- `limitation.tex:20` repeats that the representative BEIR subset is still completing

Current project source-of-truth confirms this is still not complete:

- `/Users/hangli/projects/llm-rankers/EXPERIMENT_PLAN.md:873-878`
- `/Users/hangli/projects/llm-rankers/EXPERIMENT_PLAN.md:1178`
- `/Users/hangli/projects/llm-rankers/EXPERIMENT_PLAN.md:1203`

Important detail:

- `/Users/hangli/projects/llm-rankers/EXPERIMENT_PLAN.md:876` still marks `Qwen/Qwen3.5-9B` for the representative BEIR subset as `TODO: Not Started Yet`

Implication:

- This is fine for an internal draft.
- It is **not fine** for an ARR submission if the manuscript still relies on “camera-ready” completion to finish the generalization story.
- Either:
  - finish BEIR and write it as completed work, or
  - remove all BEIR-forward-looking language and position the paper as a completed TREC-DL-only study

### 4. The current Limitations section violates ARR’s “no new content there” rule

Official ARR CFP says the `Limitations` section:

- must be before references
- does not count toward page limit
- **must not introduce new methods, analysis, or results**

Relevant official source:

- ARR CFP `Limitations`: <https://aclrollingreview.org/cfp>

Current draft problems:

- `limitation.tex:20` introduces in-progress BEIR evaluation that is not part of the completed main-paper evidence
- `limitation.tex:22` introduces the implemented-but-unevaluated refinement variants as a quasi-method package

Implication:

- This is exactly the sort of thing ARR explicitly warns against.
- The limitations section should discuss only limitations of results already established in the main paper.

### 5. Submission-strategy blocker: ARR forbids overlapping concurrent submissions

Official ARR CFP says:

- ARR precludes multiple submissions
- ARR will not consider papers under review elsewhere
- ARR will not consider work that overlaps significantly in content or results with work that will be or has been published elsewhere

Relevant official source:

- ARR CFP `Multiple Submission Policy`: <https://aclrollingreview.org/cfp>

Why this matters here:

- the ICTIR and EMNLP versions currently have the **same title**
  - ICTIR: `/Users/hangli/projects/llm-rankers/research_pipeline_setwise/ICTIR/main.tex:41`
  - EMNLP: `/Users/hangli/projects/llm-rankers/research_pipeline_setwise/EMNLP/latex/main.tex:61`
- the content overlap is extremely high: same methods, same datasets, same main tables, same narrative, same refinement package

Implication:

- If the ICTIR paper is currently under review, you should assume the current EMNLP/ARR version is **not eligible** for simultaneous submission.
- If the ICTIR version is later accepted/published, a later ARR version would need to be a **materially distinct paper**, not just a longer rewrite of the same core contribution.
- The project docs already implicitly recognized this sequencing issue by framing ARR as “later” rather than parallel.

## P1: Major Narrative / Scientific Issues

### 6. The paper is still trying to be too many papers at once

The draft currently tries to be:

- a three-method contribution paper
- a new-sorting-algorithm paper
- an asymmetry / mechanism paper
- a future-method roadmap paper

The results do not support that breadth.

The repo’s own best framing is narrower:

- `PAPER_PLAN.md:20` says the strongest narrative is a mechanism-and-analysis story rather than “three equally successful bidirectional methods”
- `RESULTS_REVIEW_V1.md:118-128` argues that the strongest novelty framing is directional asymmetry and joint elicitation, not sorting novelty alone

Concrete draft symptoms:

- `introduction.tex:86-94` lists six contributions, including the refinement package and open-source implementation
- `algorithms.tex:81-92` devotes real main-paper space to unevaluated refinement variants
- `analysis.tex:128` uses the Pareto section partly to motivate future variants rather than only explain the observed results

Recommendation:

- Recenter the paper on:
  - directional asymmetry
  - the failure of standalone worst-selection
  - the behavioral difference between standalone and joint elicitation
  - `DualEnd` as the one quality-first positive family
- Demote:
  - `BottomUp` to a diagnostic negative result
  - `BiDir` to a downstream failure mode
  - Selective / bias-aware / same-call variants to future work or supplement

### 7. The refinement package is conceptually interesting but weakens this draft

Current project planning treats the refinement package as the **next** step, not the completed current step:

- `PAPER_PLAN.md:75`
- `RESULTS_REVIEW_V1.md:191-220`
- `FINDINGS.md:13-36`

But the EMNLP draft already includes it in the contribution structure:

- `introduction.tex:92`
- `algorithms.tex:81-92`
- `conclusion.tex:16`

Why this hurts:

- It asks the reviewer to credit unevaluated ideas.
- It dilutes the cleaner asymmetry story.
- It makes the paper read as “paper plus roadmap” rather than “paper with completed evidence”.

Recommendation:

- For an ARR/EMNLP version, either:
  - fully evaluate one refinement and make it part of the new paper, or
  - remove the refinement package from the main contributions and keep only a short future-work pointer

### 8. The current evidence package still fits ICTIR better than EMNLP Long

This is not because EMNLP forbids IR papers. ARR has an `Information Retrieval and Text Mining` area, so the venue is possible in principle.

The problem is the evidence package:

- only one Bonferroni-significant `DualEnd` gain
- large efficiency cost
- incomplete generalization story
- method novelty tied heavily to sorting adaptations

This is exactly the concern already documented in:

- `RESULTS_REVIEW_V1.md:37-47`
- `RESULTS_REVIEW_V1.md:91-145`

Recommendation:

- If you want this to become an EMNLP / ARR paper, the next version should be either:
  - a **stronger completed method paper** with one evaluated refinement and finished BEIR, or
  - a **cleaner phenomenon paper** where the central claim is comparative-judgment asymmetry in LLM reranking, with the method serving primarily as the probe

## P1: Section-by-Section Audit

### Introduction

Strengths:

- The introduction already contains the right conservative language in places.
- `introduction.tex:74-84` is much closer to the strongest paper story than the older “three equally promising directions” framing.

Problems:

- `introduction.tex:88-93` still overstates the paper’s contribution spread.
- The refinement package is being counted as a contribution before it has empirical support.

Recommendation:

- Keep the asymmetry result and conservative statistical framing.
- Cut the refinement package from the contribution list unless it becomes evaluated.

### Related Work

Strengths:

- The setwise / pairwise / listwise landscape is covered.

Problems:

- The paper may still feel too IR-local for EMNLP unless the comparative-judgment / prompt-behavior angle is emphasized as an NLP contribution.
- A few citations are recent preprints / technical reports and need careful bibliography hygiene.

Recommendation:

- Strengthen the “LLM comparative judgment behavior” angle in the related-work bridge, not just “sorting in IR”.

### Methodology

Strengths:

- Clear problem setup.
- The core dual-end formulation is easy to follow.

Problems:

- The dual-end equation is one of the sources of layout overflow.
- Some implementation details could move to appendix once the paper is compressed.

Recommendation:

- Keep the problem formulation.
- Move low-level parsing details and mode-specific implementation details out of the main text if needed for page reduction.

### Algorithms

Strengths:

- Cocktail and selection are described concretely.

Problems:

- The section expands into a refinement-roadmap section.
- For the current evidence package, the paper does not need three extra variant descriptions in the main body.

Recommendation:

- If unevaluated, move the refinement package to appendix / future work.
- Keep only the two algorithms that are directly tied to evaluated results.

### Experimental Setup

Strengths:

- Good model-family coverage.
- Proper significance testing is a major strength.

Problems:

- `experiment_setup.tex:16` directly advertises in-progress BEIR.
- `experiment_setup.tex:77` is a long implementation paragraph and one source of layout stress.

Recommendation:

- Either finish BEIR or remove it from the setup narrative.
- Compress implementation detail into appendix / supplement.

### Results

Strengths:

- The results section is coherent and conservative.
- The negative results are integrated well.

Problems:

- The full main table is too large for the current layout and contributes to float pressure.
- The section still spends pages on more tables than an 8-page ARR paper can realistically afford.

Recommendation:

- For an ARR-ready version:
  - keep one main headline table
  - move either MAP@100 or some model blocks to appendix
  - keep only the ablations that directly support the paper’s central claim

### Analysis

Strengths:

- This is the strongest section in the paper.
- The position-bias and agreement analyses are the most EMNLP-friendly parts of the current story.

Problems:

- The section is too long for an 8-page paper.
- It partially turns into a prospectus for future refinements.

Recommendation:

- Promote the strongest two analysis lenses into the core paper:
  - position bias / dual-worst reversal
  - agreement / “same-call signal vs independent fusion” interpretation
- Move some of:
  - difficulty stratification
  - help/hurt table
  - full Pareto detail
  - tournament discussion
  to appendix

### Conclusion and Limitations

Strengths:

- The conclusion is cautious and honest about statistical fragility.

Problems:

- The future-work list is broad and partly about work that the current paper already foregrounds too much.
- The limitations section introduces ongoing and future material in a way ARR explicitly discourages.

Recommendation:

- Keep the current conservative conclusion.
- Rewrite limitations to describe only limitations of the completed TREC-based evidence.

## P1: Concrete Formatting Problems

### A. The “middle-of-page” line numbers are expected

This is **not** a formatting bug.

Cause:

- `main.tex:5` uses `\usepackage[review]{acl}`
- ACL review mode uses the standard review ruler
- in a two-column paper that ruler appears in:
  - the left outer margin
  - the center gutter between the two columns

Implication:

- Do **not** modify `acl.sty` or try to remove / move these line numbers for the submission PDF.
- If you want a more readable internal PDF, compile a local `preprint` or `final` copy separately.

### B. Genuine layout violations from the current build

The rebuilt `main.log` contains:

- multiple `Overfull \hbox` warnings larger than 5pt
- one `Float too large for page by 10.69498pt`

The most important ones are:

1. `introduction.tex:26-72`
   - overview TikZ figure / caption
   - log: overfull by about `44.96pt`
   - visible symptom: the figure sits too close to the gutter ruler
   - recommended fix: either scale the TikZ picture to `\columnwidth` or convert it to a `figure*` spanning both columns

2. `methodology.tex:67-70`
   - displayed dual-end equation
   - log: overfull by about `13.73pt`
   - recommended fix: split it into a 2-line `aligned` display

3. `experiment_setup.tex:57-75`
   - configuration table
   - log: overfull by about `45.06pt`
   - recommended fix: abbreviate headers / method names, reduce column count, or use a width-managed table environment

4. `results.tex:18-120`
   - main `table*`
   - log: `Float too large for page by 10.69498pt`
   - recommended fix: split the table or move one metric / some model groups to appendix

5. `results.tex:157-174`
   - efficiency table
   - log: overfull by about `48.17pt`
   - recommended fix: shorten theory formulas / labels, or use a narrower presentation

6. `results.tex:204-218`
   - `num_child` ablation table
   - log: overfull by about `13.56pt`

7. `results.tex:224-239`
   - passage-length table
   - log: overfull by about `5.88pt`

8. `analysis.tex:47-59`
   - difficulty table
   - log: overfull by about `17.68pt`

9. `analysis.tex:105-128`
   - Pareto table + accompanying lines
   - log: overfull by about `41.94pt`

Official ACL formatting rule relevant here:

- “All text except for page numbers must fit within the margins.”

Relevant official source:

- ACLPUB formatting: <https://acl-org.github.io/ACLPUB/formatting.html>

### C. Bibliography formatting is currently sloppy

The rendered references contain malformed author strings such as:

- “Aman Agrawal and 1 others”
- “Oren Wisznia and 1 others”

Cause:

- bibliography entries use `author={... and others}` in a way that renders badly with the current BibTeX style

Relevant entries:

- `reference.bib:90-101`

Recommendation:

- replace these with full author lists if possible
- otherwise fix the BibTeX author formatting so the rendered output becomes standard ACL-style `et al.`

### D. The anonymous code link is not submission-ready

Current draft:

- `introduction.tex:93` uses `\url{https://anonymous}`

Official ARR guidance says:

- supplementary repositories must be anonymized
- they should be genuinely usable / reviewable
- `Anonymous GitHub`-style solutions are acceptable

Recommendation:

- replace the placeholder with a real anonymous repository link or submit a proper anonymized supplementary archive

### E. The tables are already compressed more than necessary

Examples:

- `results.tex:21-23` uses `\scriptsize`, reduced `\tabcolsep`, and tightened `\arraystretch`

This is not automatically forbidden, but the official ACL formatting guidance explicitly warns against abusing figure/table font size and spacing.

Given that the draft is currently **far over** page limits anyway, further compression would be the wrong direction. The right fix is **content reduction / table movement**, not more squeezing.

## P2: Additional Compliance / Hygiene Notes

### 9. Appendices are allowed, but should be treated as supplementary

Official ARR guidance says:

- appendices may be submitted after the references
- they do not count toward page limit
- reviewers are **not required** to consider them
- if appendix material is important for assessing correctness, it belongs in the main paper

Relevant official source:

- ARR CFP `Optional Supplementary Materials`: <https://aclrollingreview.org/cfp>

Current implication:

- Do not rely on the appendix for any claim that reviewers must accept in order to believe the main paper.

### 10. Ethical considerations are optional but worth considering

Official ARR guidance recommends an `Ethical considerations` section placed at the end if relevant.

For this paper it is not mandatory, but a short end-matter note on:

- compute cost
- reproducibility
- retrieval evaluation scope

would be reasonable if space and positioning are handled cleanly.

### 11. AI-writing / coding disclosure is policy-relevant

Official ARR guidance says that if generative AI tools were used for writing or coding beyond trivial proofreading, the use and scope must be disclosed in:

- the Responsible NLP Checklist
- the Acknowledgements section

Relevant official source:

- ARR CFP `AI Writing/Coding Assistance Policy`: <https://aclrollingreview.org/cfp>

This draft currently has no acknowledgements section. If AI tools were materially used in writing or coding, add the required disclosure before submission.

## Recommended Repositioning

If the goal is a realistic future EMNLP / ARR version, the cleanest path is:

1. **Choose a single paper identity**
   - Preferred: a paper about **directional asymmetry and joint elicitation in LLM setwise ranking**
   - Secondary: `DualEnd` as the main positive method family

2. **Remove or demote unevaluated material**
   - Move Selective / bias-aware / same-call variants out of the main contribution list unless one becomes fully evaluated

3. **Finish the generalization package or drop it**
   - Either complete the representative BEIR story or keep the paper as a completed TREC-only study

4. **Compress the paper aggressively**
   - Move lower-priority tables and detail into appendix
   - Target an actual 8-page ARR main paper rather than polishing a long internal manuscript

5. **Decide the submission strategy before more writing**
   - If ICTIR is active, check overlap / concurrent submission policy first
   - Do not assume the current EMNLP version can simply be dual-submitted

## Prioritized Action Plan

### Must fix before any ARR / EMNLP submission

- resolve the ICTIR / ARR overlap and concurrent-submission policy question
- cut the paper to ARR-compliant main-paper length
- shorten the abstract to <= 200 words
- remove “BEIR in progress / camera-ready” language
- rewrite the limitations section so it contains no new methods, analysis, or future results
- fix the overfull / oversize layout warnings
- replace the anonymous placeholder link with a real anonymized artifact
- fix malformed bibliography author rendering

### Highest-value scientific improvements

- make the paper explicitly about directional asymmetry and joint elicitation
- keep `DualEnd` central, and demote `BottomUp` / `BiDir` to negative-result roles
- either evaluate one refinement properly or remove the refinement package from the main paper
- finish BEIR only if it materially strengthens the generalization claim

## Bottom Line

This is a promising **internal EMNLP-target draft**, but not a submission-ready ARR paper.

Right now the manuscript is strongest as:

- a careful asymmetry / mechanism paper,
- with one modestly effective quality-first method family,
- plus two coherent negative results.

That is scientifically interesting and potentially publishable, but the current version still needs:

- a policy-safe submission strategy,
- a tighter identity,
- a completed evidence package,
- and a serious compression / formatting pass.
