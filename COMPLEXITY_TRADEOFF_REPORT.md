# Report: Complexity and Trade-offs of Four Tournament/Setwise Reranking Methods

## 0. Context and Notation

This report compares four LLM-based reranking strategies:

1. **Blocked Double-Ended Tournament Reranking**
2. **Top-Heavy Double-Ended Tournament Reranking**
3. **Top-m Setwise Selection with Dualend Refinement**
4. **TourRank**

The first three are proposed variants inspired by Setwise-style LLM reranking and the Dualend idea. The fourth, **TourRank**, is an existing tournament-inspired reranking method.

The Setwise paper introduces a reranking strategy where an LLM selects the most relevant document from a set of candidates, instead of comparing only two documents at a time. This allows sorting-style methods such as bubble sort and heap sort to use a larger comparison window.

TourRank uses tournament-style stagewise selection. Documents advance through tournament rounds, receive points, and are finally ranked by accumulated scores.

The methods discussed here extend these ideas in different ways.

---

## 0.1 Notation

We use the following notation:

```text
N = number of candidate documents to rerank
B = maximum number of documents allowed in one LLM prompt
G = actual group/window size, where G ≤ B
K = target ranking depth, e.g., top-10 or top-20
r = number of repeated tournaments or repeated groupings
h = number of local Dualend extraction rounds per group
m = number of documents selected from each group in Top-m Setwise
F = size of the finalist pool after pruning/filtering
q = number of groups or blocks = ceil(N / G)
α = advancement fraction in tournament-style methods
```

Running example:

```text
N = 500
B = 50
```

The `N = 500` example is retained because it makes the comparison with
TourRank-style tournament reranking easy to see at larger rerank depths. The
project's current experimental setting is usually smaller: `EXPERIMENT_PLAN.md`
uses `hits = 100` in the main runs (`EXPERIMENT_PLAN.md:149`), while the
MaxContext design predeclares matched `hits = pool_size` baselines and warns
that archived `hits=100` runs cannot be subsetted to `hits=50`
(`IDEA_007.md:166`). Section 5 therefore includes a parallel `N = 100`
complexity table.

When useful, we also consider smaller group sizes such as:

```text
G = 20
```

A smaller group can reduce prompt length and may improve LLM reliability, even when the model technically supports 50 candidates.

---

## 0.2 Cost Metrics

We compare methods using five main cost measures:

```text
1. LLM calls
   Number of prompts sent to the LLM.

2. Candidate exposures
   Total number of document appearances across all prompts.
   Example: one prompt containing 20 documents = 20 candidate exposures.

3. Sequential depth
   Number of dependent inference steps that cannot be fully parallelized.

4. Parallel time complexity
   Wall-clock depth in dependent LLM rounds, assuming enough parallel workers.
   This differs from total LLM-call work.

5. Output type
   Whether the method produces a full ranking, a top-k ranking, or a point-based approximate ranking.
```

For LLM reranking, **candidate exposures** are often more informative than raw call count. A call with 50 long passages is much more expensive than a call with 5 short passages.

For empirical reporting, the project convention is broader than the five
conceptual measures above: report prompt tokens, completion tokens, total
tokens, parse-fallback rate, and wall-clock latency alongside LLM calls and
candidate exposures. Lower call count should not be described as token
efficiency unless the token-axis measurements also support it.

When this report says **time complexity**, it means parallel wall-clock depth
under the idealized assumption that independent LLM calls can be launched at the
same time. With no parallelism, time is proportional to the total number of LLM
calls and candidate exposures.

---

## 0.3 Relationship to the Project's MaxContext Family

The repository already contains a working MaxContext family:
`MaxContextDualEndSetwiseLlmRanker`, `MaxContextTopDownSetwiseLlmRanker`, and
`MaxContextBottomUpSetwiseLlmRanker` (`llmrankers/setwise_extended.py:1199-1996`).
`PIVOT_PLAN.md` positions this family as the project's main contribution: a
whole-pool Qwen prompt with `pool_size ≤ 50`, including MaxContext-DualEnd,
MaxContext-TopDown, and MaxContext-BottomUp (`PIVOT_PLAN.md:19-23`).

The three proposed methods in this report should therefore be read as
extensions or wrappers around the existing project primitives, not as
competitors that ignore them:

1. **Blocked Dualend vs. MaxContext-DualEnd.** If `N ≤ 50` and the whole pool
   fits in one prompt, MaxContext-DualEnd is the single-block special case of
   Blocked Double-Ended Tournament Reranking: the local block sort is the whole
   reranking problem and the multi-block merge disappears. For `N > 50`,
   Blocked Dualend adds a cross-block merge that MaxContext currently does not
   implement.
2. **Top-Heavy Dualend refinement.** Stage 3 should use MaxContext-DualEnd as
   the recommended refinement primitive whenever a finalist shard has
   `F ≤ 50`.
3. **Top-m + Dualend refinement.** Stage 3 should use the same
   MaxContext-DualEnd refinement primitive after Top-m finalist construction.

Because `IDEA_007.md` explicitly leaves `pool_size > 50` out of scope
(`IDEA_007.md:223-228`), Methods 2 and 3 use a sharded-then-merged refinement
when `F > 50`: split finalists into shards of at most 50, refine each shard
with MaxContext-DualEnd, then merge shard heads and tails with the same labeled
head/tail validation used by Method 1. This is a refinement implementation
choice inside the existing methods, not a fourth proposed method.

## 0.4 Model-Family Scope Caveat

All three proposed methods rely on a Dualend primitive that asks for best and
worst documents in the same call. In the current codebase, true joint
best+worst elicitation is effectively Qwen-generation-only. The T5 generation
path and explicit `--scoring likelihood` path score a best-only prompt and use
argmax/argmin as a proxy for best/worst (`llmrankers/setwise_extended.py:538-620`;
`PIVOT_PLAN.md:104-117`). The MaxContext constructors also enforce Qwen3 /
Qwen3.5 through `MAXCONTEXT_ALLOWED_MODEL_TYPES`
(`llmrankers/setwise_extended.py:22`, `llmrankers/setwise_extended.py:1220-1227`).

Thus, unless the implementation is extended and revalidated for additional
causal model families, the three proposed methods should be scoped as
Qwen-generation methods. Results from T5 or likelihood scoring would test a
best-only proxy, not the same Dualend mechanism.

---

# 1. Blocked Double-Ended Tournament Reranking

## 1.1 High-Level Idea

**Blocked Double-Ended Tournament Reranking** is the most sorting-like method among the proposed variants.

The LLM operation is:

```text
Given a group of candidates, return:
1. the best document in the group
2. the worst document in the group
```

This differs from standard Setwise selection, where the model selects only the best document from a set. Here, each LLM call produces two ranking decisions: one from the top and one from the bottom.

For `N = 500` and `B = 50`, we split the candidates into:

```text
500 / 50 = 10 blocks
```

The method has two phases:

```text
Phase 1: Sort each block locally using repeated best/worst extraction.
Phase 2: Merge the sorted blocks using head/tail comparison.
```

This is tournament-inspired, but it is not classical Tournament Sort. Classical Tournament Sort uses pairwise comparisons and a tournament tree. This method uses a stronger groupwise best/worst selection operation.

---

## 1.2 Algorithm

### Phase 1: Local Block Sorting

Split the candidates into blocks:

```text
Block 1: documents 1–50
Block 2: documents 51–100
...
Block 10: documents 451–500
```

For each block, repeatedly ask the LLM:

```text
Given these remaining candidates, return the most relevant and least relevant document.
```

For one block of 50:

```text
Call 1: choose local rank 1 and local rank 50
Call 2: choose local rank 2 and local rank 49
Call 3: choose local rank 3 and local rank 48
...
Call 25: choose local rank 25 and local rank 26
```

Each 50-document block requires:

```text
50 / 2 = 25 calls
```

For 10 blocks:

```text
10 × 25 = 250 calls
```

These 10 block sorts are independent. With enough parallel workers, the 10
blocks can be sorted at the same time:

```text
Total local work: 10 × 25 = 250 calls
Parallel wall-clock depth: 25 dependent rounds
```

So the local phase has `N / 2` total LLM calls, but only `B / 2` sequential
depth when block-level parallelism is available.

At the end of this phase, we have 10 internally sorted lists.

---

### Phase 1 Retrieval Sanity Check

This method has no first-stage retrieval safety net after the initial candidate
set is formed. For IR use, each local block sort should therefore log a sanity
diagnostic against the original retrieval order. If a block head is placed far
below many other documents from the same block by the first-stage retriever, or
if a block tail was originally near the top of the block, the run should flag
that block as a suspicious local sort rather than silently treating it as
equally reliable evidence. This diagnostic does not repair the ranking; it
positions Method 1 as a pure LLM-sort baseline whose early local mistakes can
propagate through the merge.

---

### Phase 2: Multiway Double-Ended Merge

For each sorted block, maintain:

```text
head = current best remaining document in the block
tail = current worst remaining document in the block
```

At each merge step, send all current heads and tails to the LLM with structural
labels that distinguish heads from tails:

```text
[H1] head_1
[T1] tail_1
[H2] head_2
[T2] tail_2
...
[H10] head_10
[T10] tail_10
```

This contains at most:

```text
10 heads + 10 tails = 20 candidates
```

The LLM returns:

```text
global best among the exposed candidates
global worst among the exposed candidates
```

The parser must validate the label class:

```text
global best must be one of the H* labels
global worst must be one of the T* labels
```

If the model returns a tail label as the global best, or a head label as the
global worst, the call is invalid. The implementation should reprompt with the
same candidates or abort the query under strict parsing. Without this
head/tail validation, a merge call could place `tail_3` ahead of `head_3`,
contradicting the Phase 1 ordering that established `head_3 > tail_3`.

Then:

```text
global best  → front of final ranking
global worst → back of final ranking
```

Remove these two documents from their corresponding blocks and update the affected heads/tails.

Since each merge call places two documents, merging 500 documents requires approximately:

```text
500 / 2 = 250 calls
```

Unlike local block sorting, the exact head/tail merge is mostly sequential. The
next merge prompt depends on which block supplied the previous global best and
worst, because only then do we know which block heads/tails should be exposed
next.

Therefore, the 250 merge calls are total work and also approximately 250
dependent merge steps in the exact algorithm. They cannot be launched as 25
independent merge calls without changing the algorithm to a more approximate or
speculative merge procedure.

---

## 1.3 Complexity

Assume:

```text
N = 500
B = 50
q = N / B = 10
```

### LLM-Call Complexity

This is total LLM work, not necessarily wall-clock depth under parallel
execution.

Local block sorting:

```text
Each block: B / 2 calls
Number of blocks: N / B

Total local calls:
(N / B) × (B / 2) = N / 2
```

Merge phase:

```text
N / 2 calls
```

Total:

```text
N / 2 + N / 2 = N
```

So the LLM-call complexity is:

```text
O(N)
```

For `N = 500`:

```text
500 calls
```

The parallelism changes latency, not total work:

```text
Local block sorting:
  total calls = N / 2
  parallel depth = B / 2

Exact head/tail merge:
  total calls = N / 2
  parallel depth ≈ N / 2
```

---

### Candidate-Exposure Complexity

For one block of size `B`, the model sees:

```text
B + (B - 2) + (B - 4) + ... + 2
```

For even `B`, this equals:

```text
(B / 2) × (B / 2 + 1)
```

For `B = 50`:

```text
25 × 26 = 650
```

For 10 blocks:

```text
10 × 650 = 6,500 candidate exposures
```

For merging:

```text
N / 2 merge calls
Each merge call sees up to 2q candidates
```

So:

```text
(N / 2) × 2q = Nq = N × (N / B)
```

For `N = 500`, `B = 50`, `q = 10`:

```text
500 × 10 = 5,000 candidate exposures
```

Total:

```text
6,500 + 5,000 = 11,500 candidate exposures
```

Asymptotically:

```text
Candidate exposures = O(NB + N² / B)
```

---

### Sequential Depth

Block sorting can be parallelized across blocks, but each block has internal sequential dependency:

```text
B / 2 sequential steps per block
```

For `N = 500`, `B = 50`, and `q = 10`, this means the 10 local block sorts can
run simultaneously and finish after about 25 dependent rounds, assuming enough
LLM workers.

The merge phase is mostly sequential:

```text
N / 2 sequential merge steps
```

So the practical sequential depth is:

```text
O(N)
```

For `N = 500`:

```text
~25 local steps + 250 merge steps
≈ 275 sequential steps
```

This is why Blocked Double-Ended Tournament Reranking is partially
parallelizable but still latency-limited by the exact merge. TourRank has a
parallelism advantage because groups within a tournament stage, and different
tournaments, can be run independently; the exact Blocked Double-Ended merge does
not have the same independence.

### Position-Bias Diagnostics

Blocked Dualend uses large group prompts in both local sorting and head/tail
merge. Existing project evidence shows that joint best+worst prompting changes
position-bias profiles: `dual_best` is flatter than single-best, while
`dual_worst` shows a primacy reversal under the current window-4 analysis
(`research-wiki/claims/C5_joint_changes_bias.md:15-22`). This claim does not
automatically transfer to larger windows or to head/tail merge prompts.

Any evaluation of this method should therefore log:

1. within-prompt order randomization for local block prompts and merge prompts
2. selected-position frequency for both best and worst labels
3. forward, reverse, and random ordering ablations at the primary configuration

This mirrors the MaxContext order-pilot discipline in `PIVOT_PLAN.md:146-158`
and the separate position-bias analysis path described in
`IDEA_007.md:187-193`.

---

### Time Complexity

Serial execution time is proportional to the total LLM work:

```text
T_serial = O(N)
```

With enough workers, local block sorting contributes only `O(B)` dependent
rounds, but the exact merge contributes `O(N)` dependent rounds:

```text
T_parallel = O(B + N) = O(N)
```

For the running example:

```text
T_parallel ≈ 25 local-sort rounds + 250 exact-merge rounds
```

---

## 1.4 Strengths

1. Produces a full ordered ranking of all `N` candidates.
2. Uses both positive and negative evidence.
3. More informative than single-winner Setwise selection.
4. Conceptually simple.
5. Easy to implement with any LLM that can return document IDs.
6. Useful as a high-fidelity full-ranking baseline.

The main advantage is that this method explicitly constructs a full ordering rather than assigning coarse points.

---

## 1.5 Weaknesses

1. Expensive for top-k IR reranking.
2. Mostly sequential during the merge phase.
3. Wastes effort sorting the middle and tail.
4. Sensitive to early local mistakes.
5. Depends on context-invariant judgments.
6. Less competitive when only `nDCG@10` or `nDCG@20` matters.

For top-heavy metrics, fully sorting all 500 documents is usually unnecessary.

---

## 1.6 Best Use Case

This method is best when:

1. A full ranking is required.
2. The reranking depth is not too large.
3. The model can reliably select both best and worst documents.
4. Latency is less important than ordering fidelity.

It is not ideal when the goal is only to maximize top-10 ranking quality under a limited inference budget.

---

# 2. Top-Heavy Double-Ended Tournament Reranking

## 2.1 High-Level Idea

**Top-Heavy Double-Ended Tournament Reranking** avoids fully sorting all documents.

Instead, it uses shallow best/worst extraction to construct a compact finalist pool, then applies more careful Dualend refinement only to that finalist pool.

The core idea is:

```text
Do not sort the whole list.
Stage 1 builds finalist recall around the retriever's anchors.
Stage 3 concentrates LLM budget on head ordering inside the finalist pool.
```

This is motivated by top-heavy IR metrics such as:

```text
nDCG@10
nDCG@20
MRR@10
```

A perfect ordering of ranks 200–500 barely matters, but better ordering among ranks 1–20 can substantially improve effectiveness.

---

## 2.2 Algorithm

The method has three stages:

```text
Stage 1: Repeated shallow Dualend extraction.
Stage 2: Construct a finalist pool.
Stage 3: Refine the finalist pool.
```

---

### Stage 1: Repeated Shallow Dualend Extraction

For each repetition:

```text
1. Split the N candidates into groups of size B or G.
2. Use stratified grouping so strong initial candidates are spread across groups.
3. Within each group, perform only h rounds of Dualend extraction.
```

Example:

```text
N = 500
B = 50
r = 3 repeated groupings
h = 3 extraction rounds per group
```

Each repetition has:

```text
500 / 50 = 10 groups
```

Each group performs:

```text
Call 1: local best and local worst
Call 2: second local best and second local worst
Call 3: third local best and third local worst
```

So each repetition requires:

```text
10 × 3 = 30 calls
```

Across 3 repetitions:

```text
3 × 30 = 90 calls
```

The repeated stratification must be genuinely different across repetitions.
Use rank-tiered shuffling within retrieval tiers and re-seed the shuffle for
each repetition. If the grouping is deterministic, `r = 3` repeats the same
evidence three times rather than reducing grouping variance.

The local winners are copied into a local-best candidate set. The local losers
provide tail evidence: they can suppress survivor-only candidates, but they do
not remove documents protected by the original top-`A` anchor set, strong
first-stage retrieval scores, or local-best extraction in another grouping.
`tail_marked` is exclusion evidence scoped to the survivor-only subset; it does
not remove anchors, high-retrieval, or local-best.

---

### Stage 2: Construct Finalist Pool

The finalist pool is constructed by set union and deterministic trimming, not
by vote aggregation. It can include:

1. Documents selected as local best during shallow Dualend extraction.
2. Documents that survive the shallow extraction rounds inside their groups and are not tail-marked.
3. Original top-`A` documents from the first-stage retriever.
4. Documents with high initial retrieval scores.

Example:

```text
A = 40 original top documents
r = 3
h = 3
local winners ≤ 3 × 3 × 10 = 90
```

After deduplication:

```text
F ≈ 100–130 finalists
```

The original top-`A` documents act as a safety net. If the LLM misses a relevant
document during shallow grouping, the original retriever can still keep it in
the finalist pool. If the union is larger than `F`, trim by a deterministic
priority rule: anchors first, then high-retrieval documents, then local-best
candidates, then survivor-only candidates, with original retrieval rank as the
tie-breaker.

---

### Finalist-Pool Recall Diagnostics

Stage 2 is an irreversible pruning step: a relevant document excluded from the
finalist pool cannot be recovered by Stage 3. Any empirical evaluation should
therefore report:

1. relevant-document recall inside the finalist pool, using qrels as an oracle
2. the top-`K` upper bound after pruning, i.e. the best achievable `nDCG@K` if
   Stage 3 perfectly ordered only the retained finalists
3. recall curves as a function of `A`, `F`, `r`, and `h`

These diagnostics should be reported before final ranking quality. Otherwise a
weak Stage 3 refiner and a low-recall finalist pool can be hard to distinguish.

---

### Stage 3: Dualend Finalist Refinement

Now apply more careful reranking only to the finalist pool. The recommended
primitive is MaxContext-DualEnd when the active pool has at most 50 documents.
For larger finalist pools, use sharded MaxContext-DualEnd refinement.

Example:

```text
F ≈ 120
Split into 3 shards of 40
Run MaxContext-DualEnd refinement inside each shard
Merge shard heads/tails with labeled head/tail validation
Produce final top-10 or top-20
```

This commits the `F > 50` case to a concrete refinement path while respecting
the current MaxContext cap. For `F ≈ 120`, the local shard refinements cost
about:

```text
3 × floor(40 / 2) = 60 Dualend calls
```

The final shard merge adds roughly another 10-20 dependent calls when only the
top-`K` prefix is needed. Cost accounting should also log any deterministic
two-document BM25 bypass used by the refiner as a separate axis. A smaller
`F ≤ 50` pool can use one MaxContext-DualEnd refinement directly, with cost
approximately `floor(F / 2)` Dualend calls plus the same explicit bypass
accounting if a two-document endgame is enabled.

---

### Position-Bias Diagnostics

Top-Heavy Dualend uses large-window joint prompts during shallow extraction and
again during finalist refinement. Because `claim:C5` shows that joint prompting
changes position bias at window size 4
(`research-wiki/claims/C5_joint_changes_bias.md:15-22`), the primary evaluation
must include order diagnostics rather than assuming neutrality at `B = 50`.

Required logs are:

1. selected-position frequencies for local best and local worst within each
   shallow group
2. selected-position frequencies during MaxContext finalist refinement
3. forward, reverse, and random ordering ablations at the primary configuration
4. parse-fallback location distribution for any strict or repaired parser path

This is the same reporting discipline as the MaxContext Phase 2 order pilot in
`PIVOT_PLAN.md:146-158` and `PIVOT_PLAN.md:441-446`.

---

## 2.3 Complexity

Assume:

```text
N = 500
B = 50
r = 3
h = 3
F ≈ 100–130
```

Because the current MaxContext implementation treats `pool_size > 50` as out of
scope (`IDEA_007.md:223-228`; `PIVOT_PLAN.md:473`), the refinement cost below
assumes sharded MaxContext-DualEnd rather than a single `F = 120` prompt.

### Stage 1 LLM Calls

Each repetition:

```text
ceil(N / B) × h
```

Across `r` repetitions:

```text
r × ceil(N / B) × h
```

For the example:

```text
3 × 10 × 3 = 90 calls
```

The Stage 1 calls are highly parallelizable. Within a single group, round 2
depends on the documents removed in round 1, so each group has `h` dependent
rounds. But different groups are independent, and repeated groupings can also be
launched together if the groupings are predetermined.

For the example:

```text
Total Stage 1 work: 3 × 10 × 3 = 90 calls
Parallel Stage 1 depth: h = 3 rounds
```

If the repeated groupings are intentionally adaptive rather than predetermined,
the Stage 1 depth becomes `r × h`; that is a different operating mode.

Asymptotically:

```text
O(rhN / B)
```

---

### Stage 1 Candidate Exposures

Within one group, `h` rounds expose:

```text
B + (B - 2) + (B - 4) + ... + (B - 2(h - 1))
```

This equals:

```text
hB - h(h - 1)
```

For `B = 50`, `h = 3`:

```text
3 × 50 - 3 × 2 = 144
```

For 10 groups and 3 repetitions:

```text
3 × 10 × 144 = 4,320 candidate exposures
```

Asymptotically:

```text
O(rhN)
```

when `h` is much smaller than `B`.

---

### Finalist Refinement Cost

Let:

```text
R_calls(F, K) = number of calls used to refine F finalists into top K
R_exp(F, K)   = candidate exposures used during refinement
R_time(F, K)  = dependent LLM rounds used during refinement
```

For `F ≈ 100–130`, sharded MaxContext-DualEnd refinement may use:

```text
60–80 calls
1,300–2,000 candidate exposures
0–1 deterministic BM25 bypass per shard/refinement path, reported separately
```

So total cost is approximately:

```text
90 + 60–80 = 150–170 calls
```

Candidate exposures:

```text
4,320 + refinement
≈ 5,600–6,300 candidate exposures
```

---

### Overall Complexity

```text
LLM calls:
O(rhN / B + R_calls(F, K))

Candidate exposures:
O(rhN + R_exp(F, K))

Sequential depth:
O(h + R_time(F, K))
```

If groups and repetitions are parallelized, the shallow extraction stage has low wall-clock depth. The final refinement stage usually dominates sequential latency.

### Time Complexity

Serial execution time is proportional to total LLM work:

```text
T_serial = O(rhN / B + R_calls(F, K))
```

With enough workers and predetermined groupings, the shallow extraction stage
collapses to `h` dependent rounds:

```text
T_parallel = O(h + R_time(F, K))
```

For the running example, Stage 1 contributes about 3 dependent rounds, and the
finalist refinement contributes the remaining dependent rounds. This is why the
overall depth is closer to `~30–40` than to the `90 + 60–80` total calls.

---

## 2.4 Strengths

1. Much cheaper than full Blocked Dualend sorting.
2. Uses both positive and negative evidence.
3. Better aligned with top-heavy IR metrics.
4. Maintains a safety net through original top-`A` candidates.
5. More robust than one-pass elimination.
6. Allows budget to be concentrated around the top-k region.

The key advantage is budget allocation.

---

## 2.5 Weaknesses

1. Does not produce a full ranking.
2. Can miss relevant documents if finalist construction is too aggressive.
3. Requires tuning `r`, `h`, `A`, and `F`.
4. Final ranking depends on grouping strategy.
5. Local worst evidence may be noisy.
6. Local worst does not always mean globally bad.

A document selected as local worst in a strong group may still be globally
useful. Therefore, local-worst evidence should not override the original top-`A`
anchor set or other strong first-stage retrieval evidence.

---

## 2.6 Best Use Case

This method is best when:

1. The target metric is top-heavy, such as `nDCG@10`.
2. We want better efficiency than full sorting.
3. The model can reliably identify both strong and weak documents.
4. We want a compact finalist pool before careful reranking.

Compared with TourRank, this method is more explicitly top-focused. Compared with full Blocked Dualend, it is much more efficient.

---

# 3. Top-m Setwise Selection with Dualend Refinement

## 3.1 High-Level Idea

**Top-m Setwise Selection with Dualend Refinement** generalizes the Setwise operation.

The original Setwise operation is:

```text
Given a query and a set of documents, select the most relevant document.
```

The Top-m extension asks:

```text
Given G documents, select the top-m documents.
```

For example:

```text
G = 20
m = 5
```

The model returns the 5 documents that appear most relevant within the group.

The unselected documents should not be treated as “the worst”. They are only the **local lower subset** relative to the selected documents.

The method has two phases:

```text
Phase 1: Repeated Top-m Setwise selection to construct a finalist pool.
Phase 2: Dualend refinement over the finalist pool.
```

---

## 3.2 Why Top-m May Be Useful

A single Setwise winner gives one strong signal:

```text
document d is best in this group
```

A Top-m selection gives a denser local partition:

```text
selected documents are preferred over unselected documents
```

If `G = 20` and `m = 5`, then one call gives:

```text
5 selected documents
15 unselected documents
```

This produces at most:

```text
5 × 15 = 75 implied cross-partition preferences with shared endpoints
```

These are not 75 independent comparisons. The response carries about:

```text
log2(C(20, 5)) ≈ 13.9 bits
```

of information, realized as at most `m × (G - m)` implied cross-partition
preferences that share endpoints. They are noisy local constraints, but they
can still be useful for finalist-pool construction.

---

## 3.3 Algorithm

Example configuration:

```text
N = 500
G = 20
m = 5
r = 3
A = 30 or 40 original top-ranked candidates
F = 80–120 finalists
K = 10 or 20
```

For each repetition:

```text
1. Stratify candidates using the initial retrieval order.
2. Split candidates into groups of 20.
3. Ask the LLM to select the top 5 documents from each group.
4. Give selected documents positive votes.
5. Optionally give unselected documents small negative votes.
```

After `r` repetitions, build a finalist pool using:

1. documents with high Top-m selection votes
2. original top-`A` documents
3. possibly documents with high initial retrieval scores

This is the vote-based finalist-construction approach. Unlike Top-Heavy
Double-Ended Tournament Reranking, repeated groupings contribute accumulated
selection evidence rather than only expanding a deterministic candidate set.

Use a predeclared scoring rule for finalist selection:

```text
score(doc) =
    normalized_positive_vote_rate(doc)
  - λ × normalized_unselected_rate(doc)
  + β × normalized_first_stage_retrieval_prior(doc)
```

Default values:

```text
λ = 0.25
β = 0.10
```

Tie-breaks are deterministic: higher positive vote rate first, then better
first-stage retrieval rank, then stable document ID order. The Top-m scoring
rule should be ablated because the same vote table can produce different
finalist pools under different `λ` and `β` values.

| Hyperparameter | Default | Sensitivity values |
|---|---:|---|
| `λ` for unselected penalty | 0.25 | 0, 0.10, 0.25, 0.50 |
| `β` for retrieval prior | 0.10 | 0, 0.05, 0.10, 0.20 |

Then apply Dualend refinement only to this finalist pool. As in Method 2, use
MaxContext-DualEnd directly when `F ≤ 50`; when `F > 50`, use sharded
MaxContext-DualEnd refinement followed by a labeled head/tail merge.

---

### Finalist-Pool Recall Diagnostics

Top-m selection prunes before final refinement. A relevant document excluded
from the finalist pool is unrecoverable, even if the Dualend refiner is perfect.
Any evaluation should report:

1. relevant-document recall in the finalist pool, using qrels as oracle evidence
2. the top-`K` upper bound after pruning, assuming perfect Stage 3 ordering
3. finalist recall as a function of `m`, `r`, `A`, `F`, `λ`, and `β`
4. tie frequency in Top-m vote scores before retrieval-prior tie-breaking

These diagnostics determine whether failures come from Top-m pruning or from
the final Dualend refinement stage.

---

### Top-m Parser Strategy

The existing Setwise and Dualend parsers handle single-label and dual-label
outputs. Top-m introduces new failure modes because the model must return a
set or ordered list of `m` labels. The parser should therefore use a structured
format and strict validation.

Recommended generative format:

```json
{"selected": [1, 4, 7, 12, 18]}
```

A fixed comma-separated form such as `1, 4, 7, 12, 18` is acceptable only if the
parser is equally strict. Validation rules:

```text
length(selected) == m
all labels are integers in [1, G]
no duplicate labels
no extra free-text explanation
```

Invalid length, duplicate IDs, missing IDs, or out-of-range IDs should count as
bad parses. Out-of-range IDs are abort-on-bad-parse errors. Refusal or
non-answer text should be logged separately from formatting errors so that
parser fragility and model refusal are not conflated.

For open-source models, logit-based Top-m selection should be treated as a
first-class option: score each numeric label under the same prompt and select
the top `m` labels by probability. This avoids generation-format errors but
still requires calibration against the generative parser because it changes the
decision rule.

---

### Position-Bias Diagnostics

Top-m prompts expose 10-50 documents and ask for multiple selections. The
position-bias diagnostics should record the position of every selected label,
not only the first selected label. Required outputs are:

1. selected-position frequency by within-prompt position
2. selected-doc stability across forward, reverse, and random ordering
3. parse-fallback location distribution
4. comparison of generative Top-m versus logit-based Top-m under the same
   orderings

This should be reported alongside the MaxContext-style order pilot because
`claim:C5` is window-4 evidence and does not establish order stability for
larger Top-m prompts (`research-wiki/claims/C5_joint_changes_bias.md:15-22`;
`IDEA_007.md:187-193`).

---

## 3.4 Complexity

Assume:

```text
N = 500
G = 20
m = 5
r = 3
F ≈ 80–120
```

### Top-m Selection Calls

Each repetition uses:

```text
ceil(N / G)
```

calls.

Across `r` repetitions:

```text
r × ceil(N / G)
```

For the example:

```text
3 × ceil(500 / 20)
= 3 × 25
= 75 calls
```

These calls are almost fully parallelizable in the vote-based finalist-pool
version. Each Top-m call is a one-shot group decision; it does not depend on
another Top-m call in the same repetition. If all groupings are predetermined,
the `r × ceil(N / G)` calls can be launched together.

For the example:

```text
Total Top-m work: 3 × 25 = 75 calls
Parallel Top-m depth: 1 LLM round
```

Asymptotically:

```text
O(rN / G)
```

---

### Top-m Candidate Exposures

Each repetition exposes each document once:

```text
N
```

Across `r` repetitions:

```text
rN
```

For the example:

```text
3 × 500 = 1,500 candidate exposures
```

Asymptotically:

```text
O(rN)
```

---

### Output-Generation Cost

If the model must generate `m` document IDs per call, then the number of generated IDs is:

```text
r × ceil(N / G) × m
```

For the example:

```text
3 × 25 × 5 = 375 generated IDs
```

This matters because generation can introduce formatting errors:

1. duplicated IDs
2. missing IDs
3. invalid IDs
4. extra explanation text
5. inconsistent ordering

A logit-based variant may avoid some of these issues if the model is open-source and label probabilities can be inspected.

The parse-fallback rate should be reported per call and per selected label.

---

### Dualend Refinement Cost

After Top-m selection, refine only the finalist pool:

```text
F ≈ 80–120
```

Sharded MaxContext-DualEnd refinement may cost:

```text
50–75 calls
1,100–1,900 candidate exposures
0–1 deterministic BM25 bypass per shard/refinement path, reported separately
```

So total cost is approximately:

```text
75 + 50–75 = 125–150 calls
```

Candidate exposures:

```text
1,500 + 1,100–1,900
= 2,600–3,400 candidate exposures
```

---

### Overall Complexity

```text
LLM calls:
O(rN / G + R_calls(F, K))

Candidate exposures:
O(rN + R_exp(F, K))

Generated output IDs:
O(rmN / G)

Sequential depth:
O(1 + refinement depth), assuming groups and repetitions are parallelized
```

Let `R_time(F, K)` be the dependent LLM rounds used by the Dualend finalist
refinement. Then the main vote-based Top-m variant has:

```text
T_serial = O(rN / G + R_calls(F, K))
T_parallel = O(1 + R_time(F, K))
```

For the running example, the 75 Top-m calls can ideally finish in one LLM round;
the Dualend finalist refinement then dominates wall-clock latency. This is why
Top-m + Dualend is potentially more parallelizable than Top-Heavy Dualend.
The practical cost claim is lower sequential LLM depth and wall-clock latency,
not guaranteed token efficiency.

---

## 3.5 Aggressive Elimination Variant

There is also a more aggressive version:

```text
At each round, keep only the selected Top-m documents.
Discard the rest.
```

For example, with:

```text
G = 20
m = 10
```

each round keeps half the documents:

```text
500 → 250 → 125 → 63 → 32 → 16
```

The number of calls is approximately:

```text
ceil(500 / 20)
+ ceil(250 / 20)
+ ceil(125 / 20)
+ ceil(63 / 20)
+ ceil(32 / 20)

= 25 + 13 + 7 + 4 + 2
= 51 calls
```

Candidate exposures:

```text
500 + 250 + 125 + 63 + 32
≈ 970 candidate exposures
```

This is very efficient, but risky. A relevant document incorrectly discarded early cannot recover later.

For this reason, the vote-based finalist-pool version is safer than the aggressive elimination version.

---

## 3.6 Strengths

1. Very efficient.
2. Produces a dense local preference signal.
3. Better aligned with top-k reranking than full sorting.
4. Can be repeated with different groupings for robustness.
5. Works naturally as a finalist-pool construction method.
6. Potentially stronger efficiency-effectiveness trade-off than full Dualend sorting.
7. Allows explicit final refinement of top candidates.

The empirical question is whether this method can beat TourRank on efficiency
while maintaining strong top-k effectiveness under matched budgets.

---

## 3.7 Weaknesses

1. Top-m generation may be unreliable.
2. Selected documents are not internally ordered.
3. Unselected documents are not necessarily bad.
4. Requires careful prompt and parser design.
5. Final effectiveness depends heavily on finalist-pool recall.
6. Larger `m` improves recall but weakens discrimination.
7. Smaller `m` improves precision but risks losing relevant documents.

A major design choice is `m`.

For `G = 20`:

```text
m = 10:
high recall, weaker signal

m = 5:
balanced

m = 3:
stronger signal, higher risk

m = 1:
standard Setwise winner selection
```

---

## 3.8 Best Use Case

This method is best when:

1. We care mainly about top-10 or top-20 ranking.
2. We want low LLM-call cost.
3. We can tolerate approximate finalist selection.
4. We can use repeated groupings to reduce grouping bias.
5. We have a strong final refinement stage.

Within-selection ambiguity transfers directly into Stage 3. For `m = 5` and
`K = 10`, documents selected in the same Top-m call have no reliable internal
order unless the parser variant asks for a ranked selected list. The evaluation
should therefore report empirical tie frequency among top vote scores and
compare unordered Top-m output with a ranked-output parser variant.

**Hypothesized** (not measured): this is a promising method for a new contribution
because it extends Setwise selection in a natural way while directly addressing
the efficiency-effectiveness trade-off.

---

# 4. TourRank

## 4.1 High-Level Idea

**TourRank** is an existing tournament-inspired LLM reranking method.

It runs multiple tournaments. In each tournament, documents are selected stage by stage. Selected documents advance to later stages and receive points. After multiple tournaments, the final ranking is produced by sorting documents according to accumulated points.

TourRank is designed around three problems:

```text
1. LLMs cannot process too many documents at once.
2. LLM outputs are sensitive to input document order.
3. Strong reranking performance can be expensive.
```

TourRank addresses these problems using:

```text
1. grouped stagewise selection
2. multiple parallel tournaments
3. accumulated point aggregation
4. seeded grouping based on initial retrieval rank
5. shuffled document order within groups
```

---

## 4.2 Algorithm

For each tournament:

```text
1. Start with N candidates.
2. Divide candidates into groups.
3. In each group, select documents to advance.
4. Selected documents receive points.
5. Selected documents advance to the next stage.
6. Repeat until later tournament stages are completed.
```

After one tournament, each document has a point value depending on how far it advanced.

Because one tournament gives coarse points, TourRank runs multiple tournaments:

```text
Tournament 1 → point vector p1
Tournament 2 → point vector p2
...
Tournament r → point vector pr
```

The final score is:

```text
p_final = p1 + p2 + ... + pr
```

The final ranking is obtained by sorting documents in descending order of accumulated points.

---

## 4.3 Grouping Strategy

TourRank does not group documents naively.

It uses the initial retrieval order to distribute strong and weak candidates across groups, similar to seeding in sports tournaments. This avoids placing all strong candidates in one group and all weak candidates in another group.

It also shuffles document order within each group to reduce position bias.

This grouping strategy is important because naive grouping can create unstable or unfair tournament outcomes.

Example problem with naive grouping:

```text
Group 1 contains ranks 1–50.
Group 2 contains ranks 51–100.

A document ranked 50th in Group 1 may be eliminated,
while a weaker document ranked 51st in Group 2 may advance.
```

Seeded grouping reduces this issue.

---

## 4.4 Complexity

Let:

```text
α = fraction of documents that advance at each stage
r = number of tournaments
G = group size
```

For one tournament, the number of document exposures is:

```text
N + αN + α²N + α³N + ...
```

This is a geometric series:

```text
N / (1 - α)
```

If approximately half the documents advance each stage:

```text
α = 1/2
```

then:

```text
N / (1 - 1/2) = 2N
```

So one tournament costs about:

```text
2N candidate exposures
```

For `r` tournaments:

```text
2rN candidate exposures
```

---

### LLM Calls

If each prompt contains up to `G` documents, calls per tournament are roughly:

```text
2N / G
```

For `r` tournaments:

```text
2rN / G
```

For the running example:

```text
N = 500
G = 50
r = 10
```

Candidate exposures:

```text
2 × 10 × 500 = 10,000
```

LLM calls:

```text
10,000 / 50 = 200 calls
```

---

### Sequential Depth

Within a tournament, selection stages are sequential because stage `s + 1` depends on the winners from stage `s`.

However:

```text
1. groups within the same stage can be run in parallel
2. different tournaments can be run in parallel
```

If half the documents advance each stage, the number of stages is roughly:

```text
O(log N)
```

So the ideal parallel depth is:

```text
O(log N)
```

This is a major advantage over sorting-based methods with long sequential dependency chains.

### Time Complexity

Serial execution time is proportional to total tournament work:

```text
T_serial = O(rN / (G(1 - α)))
```

With enough workers, groups within the same stage are parallel, and the `r`
independent tournaments can also be launched together. The wall-clock depth is
therefore the number of advancement stages in one tournament:

```text
T_parallel = O(log_{1/α} N)
```

For `α = 1/2`, this is:

```text
T_parallel = O(log₂ N)
```

If the `r` tournaments must be run sequentially because of resource limits, the
parallel-time expression becomes `O(r log_{1/α} N)`.

---

## 4.5 Strengths

1. Highly parallelizable.
2. Efficient in terms of LLM calls.
3. Robustness improves with multiple tournaments.
4. Handles input-length limits through grouping.
5. Avoids full sorting cost.
6. Uses initial ranking order through seeded grouping.
7. Strong practical baseline for tournament-style LLM reranking.

TourRank is especially strong when wall-clock latency matters because many calls can be launched in parallel.

---

## 4.6 Weaknesses

1. Does not produce an explicit sorted order through direct comparison.
2. Final ranking is point-based and can be coarse.
3. Requires multiple tournaments for fine-grained scores.
4. Ties or near-ties may require extra tie-breaking.
5. Does not explicitly use negative evidence.
6. May under-resolve the exact top-10 order.

TourRank’s point system is efficient, but it is not equivalent to full sorting. Two documents can receive similar or identical accumulated points even if one is slightly better. More tournaments reduce this issue but increase cost.

---

## 4.7 Best Use Case

TourRank is best when:

1. The reranking depth is large.
2. Parallel inference is available.
3. We want a robust approximate ranking.
4. Latency matters.
5. We do not need a strict full ordering.

It is a strong baseline for any new tournament-inspired LLM reranking method.

---

# 5. Complexity Comparison for N = 500

Assume:

```text
N = 500
B = 50
```

Representative configurations:

```text
Blocked Dualend:
B = 50

Top-Heavy Dualend:
B = 50, r = 3, h = 3, F ≈ 100–130, sharded MaxContext refinement

Top-m Setwise + Dualend:
G = 20, m = 5, r = 3, F ≈ 80–120, sharded MaxContext refinement

TourRank-10:
G = 50, r = 10, α = 1/2
```

In the following table, **TourRank-style time complexity** means the number of
dependent LLM stages under ideal parallelism, matching TourRank's `O(K_stage -
1)` accounting. `K_stage` is the number of TourRank selection stages, not the
IR top-`K` cutoff.

The token columns are reporting requirements, not claims that a method is
token-efficient. Per `PIVOT_PLAN.md:350-356`, comparisons should include total
tokens and wall-clock; prompt tokens are expected to scale roughly with
candidate exposures and passage length.

| Method | Approx. LLM Calls | Candidate Exposures | Prompt Tokens | Completion Tokens | Total Tokens | Parse-Fallback Rate | Wall-Clock / Dependent Rounds | Output Type |
|---|---:|---:|---|---|---|---|---|---|
| Blocked Double-Ended Tournament Reranking | ~500 | ~11,500 | measure; scales with exposures | measure | prompt + completion | measure strict Dualend parse failures | ~25 local-sort rounds + ~250 exact-merge rounds | Full ranking |
| Top-Heavy Double-Ended Tournament Reranking | ~150–170 | ~5,600–6,300 | measure; shard prompts included | measure | prompt + completion | measure local + refinement parse failures | ~3 shallow rounds + sharded refinement depth (~30–40 total) | Top-focused ranking |
| Top-m Setwise Selection with Dualend Refinement | ~125–150 | ~2,600–3,400 | measure; Top-m + refinement prompts | measure; `m` labels per Top-m call | prompt + completion | measure Top-m parser + Dualend parse failures | ~1 Top-m round + sharded refinement depth (~25–35 total) | Top-focused ranking |
| TourRank-10 | ~200 | ~10,000 | measure | measure | prompt + completion | measure selection parse failures | ~`K_stage - 1` stages if tournaments are parallel | Point-based approximate ranking |

The most defensible efficiency framing is fewer dependent LLM calls and lower
wall-clock latency under parallel execution. Token efficiency is empirical and
must be reported rather than assumed.

## 5.1 Complexity Comparison for N = 100

For the project's common `hits = 100` setting, the large-`N` pressure for
tournament methods is weaker and MaxContext's `pool_size ≤ 50` cap matters more
directly.

Representative configurations:

```text
N = 100
B = 50
G = 20 for Top-m
F ≤ 50 for finalist refinement where possible
```

| Method | Approx. LLM Calls | Candidate Exposures | Prompt Tokens | Completion Tokens | Total Tokens | Parse-Fallback Rate | Wall-Clock / Dependent Rounds | Output Type |
|---|---:|---:|---|---|---|---|---|---|
| Blocked Double-Ended Tournament Reranking | ~100 | ~1,500 | measure | measure | prompt + completion | measure strict Dualend parse failures | ~25 local-sort rounds + ~50 exact-merge rounds | Full ranking |
| Top-Heavy Double-Ended Tournament Reranking | ~40–50 | ~1,400–1,600 | measure | measure | prompt + completion | measure local + refinement parse failures | ~3 shallow rounds + one MaxContext refinement (~25–30 total) | Top-focused ranking |
| Top-m Setwise Selection with Dualend Refinement | ~40–45 | ~900–1,100 | measure | measure; `m` labels per Top-m call | prompt + completion | measure Top-m parser + Dualend parse failures | ~1 Top-m round + one MaxContext refinement (~25–30 total) | Top-focused ranking |
| TourRank-10 | ~40 | ~2,000 | measure | measure | prompt + completion | measure selection parse failures | ~`K_stage - 1` stages if tournaments are parallel | Point-based approximate ranking |

---

# 6. Asymptotic Comparison

| Method | LLM-Call Complexity | Candidate-Exposure Complexity | Prompt Tokens | Completion Tokens | Total Tokens | Parse-Fallback Rate | Parallel Time Complexity | Main Dependency |
|---|---|---|---|---|---|---|---|---|
| Blocked Double-Ended Tournament Reranking | `O(N)` | `O(NB + N²/B)` | `O((NB + N²/B) × passage_length)` | `O(N)` generated labels | prompt + completion | measured per Dualend call | `O(B + N) = O(N)` | Local blocks parallelize; exact head/tail merge is sequential |
| Top-Heavy Double-Ended Tournament Reranking | `O(rhN/B + R_calls(F,K))` | `O(rhN + R_exp(F,K))` | `O((rhN + R_exp) × passage_length)` | `O(rhN/B + R_calls)` labels | prompt + completion | measured for shallow + refinement calls | `O(h + R_time(F,K))` with predetermined groupings; `O(rh + R_time(F,K))` if repetitions are serialized | Finalist recall and sharded refinement |
| Top-m Setwise Selection with Dualend Refinement | `O(rN/G + R_calls(F,K))` | `O(rN + R_exp(F,K))` | `O((rN + R_exp) × passage_length)` | `O(rmN/G + R_calls)` labels | prompt + completion | measured for Top-m parser + refinement parser | `O(1 + R_time(F,K))` with predetermined groupings | Top-m reliability, scoring hyperparameters, and final refinement |
| TourRank-r | `O(rN / (G(1 - α)))` | `O(rN / (1 - α))` | `O((rN/(1 - α)) × passage_length)` | depends on advancement count | prompt + completion | measured per selection call | `O(log_{1/α} N)` if tournaments are parallel; `O(r log_{1/α} N)` if serialized | Number of tournament stages |

For TourRank, the more precise expression depends on the advancement fraction `α`:

```text
Candidate exposures per tournament:
N / (1 - α)

Candidate exposures for r tournaments:
rN / (1 - α)

LLM calls:
rN / (G(1 - α))
```

When:

```text
α = 1/2
```

this becomes:

```text
Candidate exposures ≈ 2rN
LLM calls ≈ 2rN / G
```

---

# 7. Effectiveness-Efficiency Trade-offs

## 7.1 Blocked Double-Ended Tournament Reranking

This method likely has strong effectiveness if the LLM’s best/worst judgments are reliable, because it attempts to construct a full ranking.

However, it is inefficient for top-k IR evaluation because it spends substantial budget on documents that will never affect `nDCG@10`.

Best characterization:

```text
High sorting fidelity, poor top-k efficiency.
```

---

## 7.2 Top-Heavy Double-Ended Tournament Reranking

This method is a direct improvement over Blocked Dualend for IR settings.

It keeps the useful part of Dualend:

```text
best evidence + worst evidence
```

but avoids fully sorting all documents.

Best characterization:

```text
Good top-k focus, moderate complexity, uses both positive and negative evidence.
```

---

## 7.3 Top-m Setwise Selection with Dualend Refinement

**Hypothesized** (not measured): this may be the best proposed method in terms
of efficiency-effectiveness balance.

It uses a cheap Top-m selection stage to construct a finalist pool, then applies more expensive Dualend refinement only where it matters.

Best characterization:

```text
Efficient candidate reduction + high-resolution top-k refinement.
```

This method could beat TourRank on candidate exposures and may improve top-k
effectiveness if the finalist pool has high recall, but that must be measured
under matched budgets.

The key risk is whether the LLM can reliably select `m` documents from a group without format errors or unstable behavior.

---

## 7.4 TourRank

TourRank is efficient, parallel, and robust. It is a very strong baseline.

Its main limitation is that it relies on accumulated tournament points rather than explicit top-k refinement. This can make the final top-10 order less carefully resolved than a method that performs dedicated finalist reranking.

Best characterization:

```text
Strong scalable tournament baseline, but point-based rather than explicitly sorted.
```

---

# 8. Recommended Experimental Framing

## 8.1 Main Research Question

A clean research question would be:

```text
Can a top-focused groupwise selection strategy achieve better effectiveness-efficiency trade-offs than tournament point aggregation for LLM-based reranking?
```

This avoids claiming that the proposed methods are universally better than TourRank.

Instead, it focuses on the more defensible claim:

```text
For top-heavy IR metrics, we may achieve better results by spending more budget on finalist refinement rather than broad point aggregation.
```

---

## 8.2 Suggested Method Variants

For **Top-Heavy Dualend**, test:

```text
B = 50
r ∈ {1, 3, 5}
h ∈ {1, 3, 5}
F ∈ {50, 100, 150}; F > 50 uses sharded MaxContext refinement
K ∈ {10, 20}
```

For **Top-m Setwise + Dualend**, test:

```text
G ∈ {10, 20, 50}
m ∈ {1, 3, 5, 10}
r ∈ {1, 3, 5}
F ∈ {50, 100, 150}; F > 50 uses sharded MaxContext refinement
K ∈ {10, 20}
```

For **TourRank**, compare against:

```text
TourRank-1
TourRank-2
TourRank-5
TourRank-10
```

---

## 8.3 Matched-Budget Evaluation

The fairest comparison is not just method name versus method name. It should compare methods under matched inference budgets:

```text
50 calls
100 calls
150 calls
200 calls
300 calls
```

For each budget, report the same cost axes used in the project pivot:

| Category | Required fields |
|---|---|
| Quality | `nDCG@10`, `nDCG@20`, `MRR@10`, `Recall@100` if relevant |
| Work | LLM calls, candidate exposures |
| Token cost | prompt tokens, completion tokens, total tokens |
| Robustness | parse-fallback rate, bad-parse abort rate, robustness to group shuffling |
| Latency | wall-clock latency and dependent LLM rounds |

Candidate exposures are essential because methods with the same number of calls can have very different prompt sizes.
The primary cost claim should be phrased as fewer sequential LLM calls and
lower wall-clock latency, not token efficiency unless the token-axis numbers
also support it.

---

## 8.4 Important Ablations

For the proposed methods, the most important ablations are:

1. With vs. without original top-`A` anchors.
2. Stratified grouping vs. random grouping.
3. One grouping vs. repeated groupings.
4. Top-m selection only vs. Top-m + Dualend refinement.
5. For Top-m / vote-based finalist construction: positive votes only vs. positive and negative evidence.
6. Generative Top-m output vs. logit-based Top-m selection.
7. Different finalist-pool sizes.
8. Top-m scoring weights: `λ ∈ {0, 0.10, 0.25, 0.50}` and `β ∈ {0, 0.05, 0.10, 0.20}`.

For Top-m specifically:

```text
m = 1:
equivalent to single-winner Setwise-style selection

m = 3 or 5:
stronger precision signal

m = 10 from G = 20:
higher recall but weaker discrimination
```

---

# 9. Likely Ranking of Methods

## 9.1 Efficiency

```text
1. Top-m Setwise Selection with Dualend Refinement
2. Top-Heavy Double-Ended Tournament Reranking
3. TourRank-10
4. Blocked Double-Ended Tournament Reranking
```

---

## 9.2 Parallelizability

```text
1. TourRank
2. Top-m Setwise Selection with Dualend Refinement
3. Top-Heavy Double-Ended Tournament Reranking
4. Blocked Double-Ended Tournament Reranking
```

Blocked Double-Ended Tournament Reranking is not fully serial: its local block
sorting phase is parallel across blocks. It ranks last here because the exact
multiway merge remains a long dependent chain, not because the local phase lacks
parallelism.

---

## 9.3 Full-Ranking Fidelity

```text
1. Blocked Double-Ended Tournament Reranking
2. Top-Heavy Double-Ended Tournament Reranking
3. Top-m Setwise Selection with Dualend Refinement
4. TourRank
```

---

## 9.4 Likely Top-k Effectiveness Under a Fixed Moderate Budget

**Hypothesized** (not measured):

1. *Top-m Setwise Selection with Dualend Refinement*
2. *Top-Heavy Double-Ended Tournament Reranking*
3. *TourRank*
4. *Blocked Double-Ended Tournament Reranking*

This ranking is a hypothesis, not a guaranteed result. TourRank is a strong baseline, and its multiple-tournament point aggregation may be very robust. The proposed methods need empirical validation under matched inference budgets.

---

# 10. Final Takeaways

The four methods occupy different positions in the effectiveness-efficiency space.

```text
Blocked Double-Ended Tournament Reranking
= full sorting, high cost, strong ordering fidelity

Top-Heavy Double-Ended Tournament Reranking
= shallow Dualend filtering, finalist refinement, good top-k focus

Top-m Setwise Selection with Dualend Refinement
= dense groupwise selection signal, cheap finalist construction, hypothesized strong efficiency-effectiveness trade-off

TourRank
= parallel tournament point aggregation, strong baseline, robust and scalable
```

**Hypothesized** (not measured): the most promising new method is:

*Top-m Setwise Selection with Dualend Refinement*

because it combines three useful properties:

```text
1. Setwise-style efficiency:
   each prompt compares many documents.

2. Top-m density:
   each prompt returns more information than a single winner.

3. Dualend refinement:
   the final top candidates are explicitly resolved rather than only ranked by accumulated points.
```

This framing remains conditional on the MaxContext relationship described in
§0.3. Stage 3 should use the existing MaxContext-DualEnd primitive where
possible, and any `F > 50` setting should be reported as sharded MaxContext
refinement rather than as a single unsupported large-pool MaxContext call.

**Hypothesized** (not measured): a strong paper framing would be:

```text
TourRank shows that tournament-style point aggregation can make LLM reranking efficient and robust. However, point aggregation may under-resolve the top-ranked documents that matter most for top-heavy IR metrics. We therefore propose a top-focused groupwise reranking strategy that first uses Top-m Setwise selection to build a compact high-recall finalist pool, then applies Dualend refinement to explicitly resolve the top-k ranking. This design reduces inference cost while concentrating ranking capacity on the most metric-sensitive region.
```

## 10.1 Open Methodological Gaps

The following diagnostics are required before any future empirical comparison
should be treated as evidence rather than a hypothesis:

1. **Finalist-pool recall.** Methods 2 and 3 must report relevant-document
   recall in the finalist pool and the top-`K` upper bound after pruning.
2. **Position bias.** All three proposed methods must report selected-position
   frequencies, forward/reverse/random ordering ablations, and parse-fallback
   location distributions.
3. **Top-m scoring hyperparameters.** Method 3 must predeclare `λ`, retrieval
   prior weight `β`, normalization, and tie-breaks, then report sensitivity to
   those choices.

---

# 11. Pseudocode Algorithms

The pseudocode below is conceptual. `PARALLEL FOR` marks calls that can be
launched concurrently under the same idealized parallel-time model used by
TourRank.

## 11.1 Blocked Double-Ended Tournament Reranking

```text
INPUT:
  query q
  candidate documents D = [d1, ..., dN]
  block size B

HELPER:
  DUALEND(q, S):
    Ask the LLM over candidate set S.
    Return (best_doc, worst_doc).

  DUALEND_HEAD_TAIL_MERGE(q, heads, tails):
    Render a labeled prompt:
      [H1] head from block 1
      [T1] tail from block 1
      [H2] head from block 2
      [T2] tail from block 2
      ...
    Ask for:
      Best head label: H*
      Worst tail label: T*
    Parse labels and validate:
      best label starts with H
      worst label starts with T
      labels refer to currently exposed candidates
      best_doc != worst_doc
    If validation fails:
      reprompt once with the same labels, or abort under strict parsing.
    Return (best_doc, worst_doc).

  LOCAL_DUALEND_SORT(q, block):
    live = block
    top = []
    bottom = []

    while |live| > 1:
      best, worst = DUALEND(q, live)
      append best to top
      prepend worst to bottom
      remove best and worst from live

    if |live| == 1:
      append the remaining document to top

    return top + bottom

ALGORITHM:
  blocks = split D into consecutive blocks of size at most B

  PARALLEL FOR each block b in blocks:
    sorted_block[b] = LOCAL_DUALEND_SORT(q, b)

  For each sorted_block:
    head[b] = index of current best remaining document
    tail[b] = index of current worst remaining document

  front = []
  back = []

  while more than one document remains across all blocks:
    heads = []
    tails = []

    for each non-empty sorted_block b:
      add (b, sorted_block[b][head[b]]) to heads
      if head[b] != tail[b]:
        add (b, sorted_block[b][tail[b]]) to tails

    best, worst = DUALEND_HEAD_TAIL_MERGE(q, heads, tails)

    append best to front
    prepend worst to back

    remove best from its block
    remove worst from its block
    update affected head/tail pointers

  if one document remains:
    append it to front

  return front + back
```

Parallelism:

```text
Local block sorting: parallel across blocks, depth B / 2
Exact head/tail merge: mostly sequential, depth N / 2
TourRank-style time: O(B + N) = O(N)
```

---

## 11.2 Top-Heavy Double-Ended Tournament Reranking

```text
INPUT:
  query q
  candidate documents D = [d1, ..., dN]
  group size B
  repetitions r
  local Dualend rounds h
  original-anchor size A
  finalist-pool size F
  target depth K

HELPER:
  STRATIFIED_GROUPS(D, B, seed):
    Spread initially high-ranked and low-ranked documents across groups.
    Partition D into retrieval-rank tiers.
    Shuffle within each tier using seed.
    Deal shuffled tier members round-robin into groups.
    Return groups of size at most B.

  MAXCONTEXT_DUALEND_REFINE(q, pool, K):
    Requires |pool| <= 50 under the current implementation.
    Run MaxContext-DualEnd over the pool.
    Cost is approximately floor(|pool| / 2) Dualend calls.
    If a deterministic two-document endgame is enabled, log one BM25 bypass
    separately from LLM calls.
    Return a top-K or top-focused ranking.

  SHARDED_MAXCONTEXT_REFINE(q, finalists, K, shard_size=40):
    if |finalists| <= 50:
      return MAXCONTEXT_DUALEND_REFINE(q, finalists, K)

    Split finalists into retrieval-stratified shards of size <= shard_size.

    PARALLEL FOR each shard s:
      local_ranked[s] = MAXCONTEXT_DUALEND_REFINE(q, shard s, K)

    merge_live = local_ranked shard lists
    Use the labeled head/tail merge validation from §11.1 to resolve the
    final top-K across shard heads and tails.
    Append unresolved shard contents by local rank and original retrieval rank.
    Return top-K or top-focused ranking.

  DUALEND_REFINE(q, finalists, K):
    return SHARDED_MAXCONTEXT_REFINE(q, finalists, K)

ALGORITHM:
  local_best_candidates = empty set
  survivor_candidates = empty set
  tail_marked = empty set

  PARALLEL FOR repetition t in {1, ..., r}:
    groups = STRATIFIED_GROUPS(D, B, seed=t)

    PARALLEL FOR each group g in groups:
      live = g

      for round j in {1, ..., h}:
        if |live| < 2:
          break

        best, worst = DUALEND(q, live)
        add best to local_best_candidates
        add worst to tail_marked

        remove best and worst from live

      add remaining live documents to survivor_candidates

  anchor_docs = original top-A documents from D
  high_retrieval_docs = documents with high first-stage retrieval scores
  survivor_only_docs =
      survivor_candidates
      minus tail_marked
      minus local_best_candidates
      minus anchor_docs
      minus high_retrieval_docs

  finalists =
      anchor_docs
      union high_retrieval_docs
      union local_best_candidates
      union survivor_only_docs

  trim finalists to size F using deterministic priority:
    1. anchor_docs first
    2. high_retrieval_docs
    3. local_best_candidates
    4. survivor_only_docs
    5. original retrieval rank as tie-breaker

  tail_marked is exclusion evidence scoped to the survivor-only subset;
  it does not remove anchors, high-retrieval, or local-best.

  refined_top = DUALEND_REFINE(q, finalists, K)

  append any unplaced finalists after refined_top using the original retrieval order
  append non-finalists after that using the original retrieval order

  return final ranking
```

Parallelism:

```text
Shallow extraction: parallel across groups and predetermined repetitions
Within each group: h dependent Dualend rounds
Finalist refinement: depends on the chosen refinement algorithm
TourRank-style time: O(h + R_time(F, K))
```

---

## 11.3 Top-m Setwise Selection with Dualend Refinement

```text
INPUT:
  query q
  candidate documents D = [d1, ..., dN]
  group size G
  selected count m
  repetitions r
  original-anchor size A
  finalist-pool size F
  target depth K

HELPER:
  TOP_M(q, S, m):
    Ask the LLM to return structured output:
      {"selected": [integer labels]}
    Validate:
      length(selected) == m
      every label is an integer in [1, |S|]
      no duplicate labels
      no extra explanation text
    If the model refuses or returns malformed output:
      log refusal_or_bad_parse
      reprompt once or abort under strict parsing.
    Return selected_docs.

  TOP_M_LOGIT(q, S, m):
    For open-source models, score numeric labels under the same prompt.
    Select the m labels with highest label probability.
    Return selected_docs.

  SHARDED_MAXCONTEXT_REFINE(q, finalists, K, shard_size=40):
    if |finalists| <= 50:
      run MaxContext-DualEnd directly
      log deterministic BM25 bypass separately if a two-document endgame is enabled
      return top-K or top-focused ranking

    Split finalists into retrieval-stratified shards of size <= shard_size.
    PARALLEL FOR each shard:
      run MaxContext-DualEnd on that shard
    Merge shard heads/tails using the labeled validation from §11.1.
    Return top-K or top-focused ranking.

  DUALEND_REFINE(q, finalists, K):
    return SHARDED_MAXCONTEXT_REFINE(q, finalists, K)

ALGORITHM:
  positive_votes = empty counter
  negative_votes = empty counter

  PARALLEL FOR repetition t in {1, ..., r}:
    groups = STRATIFIED_GROUPS(D, G, seed=t)

    PARALLEL FOR each group g in groups:
      selected = TOP_M(q, g, m)
      unselected = g - selected

      for doc in selected:
        positive_votes[doc] += 1

      for doc in unselected:
        negative_votes[doc] += optional_small_penalty

  anchor_docs = original top-A documents from D

  lambda = 0.25
  retrieval_prior_weight = 0.10

  score(doc) =
      normalized_positive_vote_rate(doc)
    - lambda * normalized_unselected_rate(doc)
    + retrieval_prior_weight * normalized_first_stage_retrieval_prior(doc)

  finalists =
      anchor_docs
      union highest-scoring documents by score(doc)

  trim finalists to size F using deterministic tie-breaks:
    1. higher positive vote rate
    2. better first-stage retrieval rank
    3. stable document ID order

  refined_top = DUALEND_REFINE(q, finalists, K)

  append any unplaced finalists after refined_top using the original retrieval order
  append non-finalists after that using the original retrieval order

  return final ranking
```

Parallelism:

```text
Top-m finalist construction: one-shot calls, parallel across all groups
Predetermined repetitions can also run in parallel
Finalist refinement: depends on the chosen refinement algorithm
TourRank-style time: O(1 + R_time(F, K))
```

---

## 11.4 TourRank

```text
INPUT:
  query q
  candidate documents D = [d1, ..., dN]
  group size G
  number of tournaments r
  number of stages K_stage
  advancement counts a_1, ..., a_(K_stage - 1)

HELPER:
  SELECT_ADVANCERS(q, S, a):
    Ask the LLM to select a documents from S to advance.
    Return selected_docs.

ALGORITHM:
  global_points = empty counter

  PARALLEL FOR tournament t in {1, ..., r}:
    active = D
    local_points = empty counter

    for stage s in {1, ..., K_stage - 1}:
      groups = SEEDED_GROUPS(active, G, tournament=t, stage=s)
      next_active = []

      PARALLEL FOR each group g in groups:
        winners = SELECT_ADVANCERS(q, g, a_s)

        for doc in winners:
          local_points[doc] += points_for_stage(s)
          add doc to next_active

      active = next_active

    for doc in local_points:
      global_points[doc] += local_points[doc]

  return documents sorted by:
    1. descending global_points
    2. optional first-stage retrieval score as tie-breaker
```

Parallelism:

```text
Within a stage: groups run in parallel
Across tournaments: tournaments run in parallel
Across stages: sequential, because stage s+1 depends on stage s winners
TourRank-style time: O(K_stage - 1)
```

---

# 13. Review Audit Trail

2026-04-29 — Two-round dialectic review (Claude + Codex gpt-5.5 + xhigh)
yielded 18 substantive findings (3 BLOCKER + 9 HIGH + 5 MEDIUM + 1 LOW);
this version addresses all of them. See
`.planning/complexity_report/AGREED_FEEDBACK.md`.
