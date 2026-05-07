# Significance Tests — Pairwise (same-sort comparisons)

## Method

- Per-query `ndcg_cut_10` values are read from the saved `.eval` files under `results/`.
- Each comparison uses a **specific** TopDown baseline against a **specific** challenger method (no family-best reduction).
- Statistical test: two-sided paired approximate randomization with `100,000` samples.
- Uncertainty: paired bootstrap 95% CI on mean delta with `20,000` resamples.
- Multiple testing: Bonferroni correction **per (grouping, dataset)** across (9 models × |challengers|) tests. Each table is its own hypothesis family.

## Summary (per grouping, per dataset)

| Grouping | Dataset | Mean delta | Positive / total | Bonferroni wins | Bonferroni losses |
|---|---|---:|---:|---:|---:|
| Bubblesort family: TD-Bubble baseline vs BU-Bubble, DE-Cocktail | DL19 | -0.0291 | 7/18 | 1 | 4 |
| Bubblesort family: TD-Bubble baseline vs BU-Bubble, DE-Cocktail | DL20 | -0.0260 | 7/18 | 1 | 4 |
| Heapsort family: TD-Heap baseline vs BU-Heap | DL19 | -0.0786 | 1/9 | 0 | 4 |
| Heapsort family: TD-Heap baseline vs BU-Heap | DL20 | -0.0946 | 0/9 | 0 | 6 |
| DualEnd vs TD-Bubble baseline | DL19 | +0.0043 | 10/18 | 2 | 0 |
| DualEnd vs TD-Bubble baseline | DL20 | +0.0007 | 10/18 | 1 | 1 |
| DualEnd vs TD-Heap baseline | DL19 | +0.0084 | 12/18 | 1 | 0 |
| DualEnd vs TD-Heap baseline | DL20 | +0.0058 | 12/18 | 0 | 0 |
| Bidirectional vs TD-Bubble baseline | DL19 | -0.0251 | 3/18 | 0 | 3 |
| Bidirectional vs TD-Bubble baseline | DL20 | -0.0239 | 3/18 | 0 | 5 |
| Bidirectional vs TD-Heap baseline | DL19 | -0.0210 | 3/18 | 0 | 1 |
| Bidirectional vs TD-Heap baseline | DL20 | -0.0188 | 2/18 | 0 | 2 |

## Bubblesort family: TD-Bubble baseline vs BU-Bubble, DE-Cocktail — DL19

| Model | Baseline | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | TD-Bubble 0.6874 | BU-Bubble 0.4571 | -0.2302 | [-0.2905, -0.1736] | 6-34-3 | <0.001 | <0.001 | sig loss |
| flan-t5-large | TD-Bubble 0.6874 | DE-Cocktail 0.6708 | -0.0165 | [-0.0401, +0.0042] | 14-21-8 | 0.156 | 1.000 | -ns |
| flan-t5-xl | TD-Bubble 0.6980 | BU-Bubble 0.6730 | -0.0250 | [-0.0520, +0.0015] | 13-21-9 | 0.076 | 1.000 | -ns |
| flan-t5-xl | TD-Bubble 0.6980 | DE-Cocktail 0.6884 | -0.0096 | [-0.0258, +0.0046] | 18-16-9 | 0.245 | 1.000 | -ns |
| flan-t5-xxl | TD-Bubble 0.7077 | BU-Bubble 0.6936 | -0.0141 | [-0.0345, +0.0056] | 15-24-4 | 0.183 | 1.000 | -ns |
| flan-t5-xxl | TD-Bubble 0.7077 | DE-Cocktail 0.7137 | +0.0060 | [-0.0085, +0.0209] | 17-18-8 | 0.436 | 1.000 | +ns |
| qwen3-4b | TD-Bubble 0.6491 | BU-Bubble 0.6305 | -0.0187 | [-0.0449, +0.0067] | 14-26-3 | 0.166 | 1.000 | -ns |
| qwen3-4b | TD-Bubble 0.6491 | DE-Cocktail 0.6796 | +0.0305 | [+0.0091, +0.0528] | 28-12-3 | 0.008 | 0.141 | +ns |
| qwen3-8b | TD-Bubble 0.6794 | BU-Bubble 0.6273 | -0.0522 | [-0.0880, -0.0181] | 14-26-3 | 0.006 | 0.106 | -ns |
| qwen3-8b | TD-Bubble 0.6794 | DE-Cocktail 0.7155 | +0.0361 | [+0.0172, +0.0548] | 31-9-3 | <0.001 | 0.013 | sig win |
| qwen3-14b | TD-Bubble 0.7455 | BU-Bubble 0.6702 | -0.0752 | [-0.1069, -0.0463] | 7-31-5 | <0.001 | <0.001 | sig loss |
| qwen3-14b | TD-Bubble 0.7455 | DE-Cocktail 0.7519 | +0.0064 | [-0.0055, +0.0190] | 17-17-9 | 0.327 | 1.000 | +ns |
| qwen3.5-4b | TD-Bubble 0.7108 | BU-Bubble 0.6120 | -0.0988 | [-0.1441, -0.0542] | 13-28-2 | <0.001 | 0.002 | sig loss |
| qwen3.5-4b | TD-Bubble 0.7108 | DE-Cocktail 0.7161 | +0.0052 | [-0.0152, +0.0253] | 20-19-4 | 0.620 | 1.000 | +ns |
| qwen3.5-9b | TD-Bubble 0.7349 | BU-Bubble 0.6712 | -0.0637 | [-0.1034, -0.0284] | 11-27-5 | <0.001 | 0.017 | sig loss |
| qwen3.5-9b | TD-Bubble 0.7349 | DE-Cocktail 0.7370 | +0.0021 | [-0.0192, +0.0250] | 20-19-4 | 0.861 | 1.000 | +ns |
| qwen3.5-27b | TD-Bubble 0.7435 | BU-Bubble 0.7336 | -0.0099 | [-0.0370, +0.0153] | 13-19-11 | 0.477 | 1.000 | -ns |
| qwen3.5-27b | TD-Bubble 0.7435 | DE-Cocktail 0.7475 | +0.0040 | [-0.0095, +0.0183] | 15-15-13 | 0.583 | 1.000 | +ns |

## Bubblesort family: TD-Bubble baseline vs BU-Bubble, DE-Cocktail — DL20

| Model | Baseline | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | TD-Bubble 0.6264 | BU-Bubble 0.4116 | -0.2148 | [-0.2825, -0.1485] | 7-45-2 | <0.001 | <0.001 | sig loss |
| flan-t5-large | TD-Bubble 0.6264 | DE-Cocktail 0.6308 | +0.0044 | [-0.0160, +0.0268] | 21-28-5 | 0.702 | 1.000 | +ns |
| flan-t5-xl | TD-Bubble 0.6868 | BU-Bubble 0.6691 | -0.0177 | [-0.0445, +0.0078] | 23-27-4 | 0.191 | 1.000 | -ns |
| flan-t5-xl | TD-Bubble 0.6868 | DE-Cocktail 0.6795 | -0.0073 | [-0.0181, +0.0028] | 19-27-8 | 0.185 | 1.000 | -ns |
| flan-t5-xxl | TD-Bubble 0.6959 | BU-Bubble 0.6811 | -0.0148 | [-0.0404, +0.0105] | 20-29-5 | 0.266 | 1.000 | -ns |
| flan-t5-xxl | TD-Bubble 0.6959 | DE-Cocktail 0.6895 | -0.0064 | [-0.0206, +0.0075] | 24-22-8 | 0.382 | 1.000 | -ns |
| qwen3-4b | TD-Bubble 0.6269 | BU-Bubble 0.5778 | -0.0491 | [-0.0768, -0.0220] | 17-34-3 | <0.001 | 0.015 | sig loss |
| qwen3-4b | TD-Bubble 0.6269 | DE-Cocktail 0.6454 | +0.0185 | [+0.0010, +0.0362] | 31-19-4 | 0.045 | 0.813 | +ns |
| qwen3-8b | TD-Bubble 0.6372 | BU-Bubble 0.5948 | -0.0424 | [-0.0768, -0.0096] | 21-29-4 | 0.016 | 0.288 | -ns |
| qwen3-8b | TD-Bubble 0.6372 | DE-Cocktail 0.6678 | +0.0306 | [+0.0123, +0.0494] | 33-16-5 | 0.002 | 0.039 | sig win |
| qwen3-14b | TD-Bubble 0.7044 | BU-Bubble 0.6395 | -0.0649 | [-0.0959, -0.0352] | 13-39-2 | <0.001 | <0.001 | sig loss |
| qwen3-14b | TD-Bubble 0.7044 | DE-Cocktail 0.7051 | +0.0007 | [-0.0152, +0.0166] | 20-26-8 | 0.934 | 1.000 | +ns |
| qwen3.5-4b | TD-Bubble 0.6713 | BU-Bubble 0.5963 | -0.0749 | [-0.1161, -0.0331] | 12-37-5 | <0.001 | 0.016 | sig loss |
| qwen3.5-4b | TD-Bubble 0.6713 | DE-Cocktail 0.6768 | +0.0055 | [-0.0145, +0.0251] | 24-24-6 | 0.596 | 1.000 | +ns |
| qwen3.5-9b | TD-Bubble 0.6925 | BU-Bubble 0.6677 | -0.0247 | [-0.0560, +0.0068] | 18-30-6 | 0.133 | 1.000 | -ns |
| qwen3.5-9b | TD-Bubble 0.6925 | DE-Cocktail 0.6984 | +0.0059 | [-0.0127, +0.0259] | 22-25-7 | 0.568 | 1.000 | +ns |
| qwen3.5-27b | TD-Bubble 0.7178 | BU-Bubble 0.7004 | -0.0174 | [-0.0416, +0.0068] | 18-28-8 | 0.165 | 1.000 | -ns |
| qwen3.5-27b | TD-Bubble 0.7178 | DE-Cocktail 0.7186 | +0.0008 | [-0.0106, +0.0128] | 17-28-9 | 0.893 | 1.000 | +ns |

## Heapsort family: TD-Heap baseline vs BU-Heap — DL19

| Model | Baseline | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | TD-Heap 0.6541 | BU-Heap 0.2888 | -0.3654 | [-0.4415, -0.2908] | 2-41-0 | <0.001 | <0.001 | sig loss |
| flan-t5-xl | TD-Heap 0.6901 | BU-Heap 0.6630 | -0.0270 | [-0.0639, +0.0100] | 12-24-7 | 0.164 | 1.000 | -ns |
| flan-t5-xxl | TD-Heap 0.6846 | BU-Heap 0.6874 | +0.0028 | [-0.0270, +0.0348] | 15-24-4 | 0.863 | 1.000 | +ns |
| qwen3-4b | TD-Heap 0.6775 | BU-Heap 0.6261 | -0.0514 | [-0.0942, -0.0086] | 14-27-2 | 0.024 | 0.214 | -ns |
| qwen3-8b | TD-Heap 0.6819 | BU-Heap 0.6431 | -0.0388 | [-0.0803, +0.0006] | 14-24-5 | 0.069 | 0.625 | -ns |
| qwen3-14b | TD-Heap 0.7447 | BU-Heap 0.6966 | -0.0482 | [-0.0824, -0.0183] | 9-25-9 | 0.002 | 0.018 | sig loss |
| qwen3.5-4b | TD-Heap 0.7087 | BU-Heap 0.6158 | -0.0929 | [-0.1404, -0.0479] | 12-28-3 | <0.001 | 0.002 | sig loss |
| qwen3.5-9b | TD-Heap 0.7329 | BU-Heap 0.6779 | -0.0549 | [-0.0831, -0.0285] | 13-27-3 | <0.001 | 0.002 | sig loss |
| qwen3.5-27b | TD-Heap 0.7449 | BU-Heap 0.7135 | -0.0314 | [-0.0611, -0.0053] | 12-23-8 | 0.031 | 0.283 | -ns |

## Heapsort family: TD-Heap baseline vs BU-Heap — DL20

| Model | Baseline | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | TD-Heap 0.6100 | BU-Heap 0.2203 | -0.3897 | [-0.4630, -0.3142] | 5-48-1 | <0.001 | <0.001 | sig loss |
| flan-t5-xl | TD-Heap 0.6680 | BU-Heap 0.6241 | -0.0440 | [-0.0780, -0.0114] | 20-30-4 | 0.014 | 0.125 | -ns |
| flan-t5-xxl | TD-Heap 0.6790 | BU-Heap 0.6625 | -0.0166 | [-0.0490, +0.0163] | 20-32-2 | 0.329 | 1.000 | -ns |
| qwen3-4b | TD-Heap 0.6488 | BU-Heap 0.5963 | -0.0525 | [-0.0936, -0.0095] | 16-38-0 | 0.020 | 0.183 | -ns |
| qwen3-8b | TD-Heap 0.6532 | BU-Heap 0.5963 | -0.0569 | [-0.0933, -0.0210] | 17-36-1 | 0.003 | 0.027 | sig loss |
| qwen3-14b | TD-Heap 0.6962 | BU-Heap 0.6323 | -0.0639 | [-0.0924, -0.0363] | 13-38-3 | <0.001 | <0.001 | sig loss |
| qwen3.5-4b | TD-Heap 0.6521 | BU-Heap 0.5550 | -0.0971 | [-0.1441, -0.0521] | 15-36-3 | <0.001 | 0.001 | sig loss |
| qwen3.5-9b | TD-Heap 0.6950 | BU-Heap 0.6319 | -0.0631 | [-0.0949, -0.0336] | 13-38-3 | <0.001 | <0.001 | sig loss |
| qwen3.5-27b | TD-Heap 0.7104 | BU-Heap 0.6426 | -0.0679 | [-0.1197, -0.0247] | 18-33-3 | 0.002 | 0.021 | sig loss |

## DualEnd vs TD-Bubble baseline — DL19

| Model | Baseline | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | TD-Bubble 0.6874 | DE-Cocktail 0.6708 | -0.0165 | [-0.0403, +0.0040] | 14-21-8 | 0.156 | 1.000 | -ns |
| flan-t5-large | TD-Bubble 0.6874 | DE-Selection 0.6420 | -0.0454 | [-0.0864, -0.0099] | 16-21-6 | 0.021 | 0.372 | -ns |
| flan-t5-xl | TD-Bubble 0.6980 | DE-Cocktail 0.6884 | -0.0096 | [-0.0258, +0.0049] | 18-16-9 | 0.243 | 1.000 | -ns |
| flan-t5-xl | TD-Bubble 0.6980 | DE-Selection 0.6792 | -0.0188 | [-0.0410, +0.0022] | 14-22-7 | 0.097 | 1.000 | -ns |
| flan-t5-xxl | TD-Bubble 0.7077 | DE-Cocktail 0.7137 | +0.0060 | [-0.0081, +0.0208] | 17-18-8 | 0.435 | 1.000 | +ns |
| flan-t5-xxl | TD-Bubble 0.7077 | DE-Selection 0.6974 | -0.0103 | [-0.0377, +0.0167] | 18-19-6 | 0.474 | 1.000 | -ns |
| qwen3-4b | TD-Bubble 0.6491 | DE-Cocktail 0.6796 | +0.0305 | [+0.0095, +0.0531] | 28-12-3 | 0.008 | 0.146 | +ns |
| qwen3-4b | TD-Bubble 0.6491 | DE-Selection 0.7220 | +0.0729 | [+0.0399, +0.1081] | 32-8-3 | <0.001 | <0.001 | sig win |
| qwen3-8b | TD-Bubble 0.6794 | DE-Cocktail 0.7155 | +0.0361 | [+0.0173, +0.0549] | 31-9-3 | <0.001 | 0.011 | sig win |
| qwen3-8b | TD-Bubble 0.6794 | DE-Selection 0.7158 | +0.0364 | [+0.0070, +0.0677] | 28-13-2 | 0.024 | 0.437 | +ns |
| qwen3-14b | TD-Bubble 0.7455 | DE-Cocktail 0.7519 | +0.0064 | [-0.0055, +0.0188] | 17-17-9 | 0.328 | 1.000 | +ns |
| qwen3-14b | TD-Bubble 0.7455 | DE-Selection 0.7475 | +0.0020 | [-0.0188, +0.0222] | 17-17-9 | 0.850 | 1.000 | +ns |
| qwen3.5-4b | TD-Bubble 0.7108 | DE-Cocktail 0.7161 | +0.0052 | [-0.0155, +0.0251] | 20-19-4 | 0.619 | 1.000 | +ns |
| qwen3.5-4b | TD-Bubble 0.7108 | DE-Selection 0.7022 | -0.0086 | [-0.0296, +0.0124] | 18-22-3 | 0.434 | 1.000 | -ns |
| qwen3.5-9b | TD-Bubble 0.7349 | DE-Cocktail 0.7370 | +0.0021 | [-0.0195, +0.0256] | 20-19-4 | 0.860 | 1.000 | +ns |
| qwen3.5-9b | TD-Bubble 0.7349 | DE-Selection 0.7309 | -0.0041 | [-0.0300, +0.0210] | 16-21-6 | 0.763 | 1.000 | -ns |
| qwen3.5-27b | TD-Bubble 0.7435 | DE-Cocktail 0.7475 | +0.0040 | [-0.0095, +0.0180] | 15-15-13 | 0.583 | 1.000 | +ns |
| qwen3.5-27b | TD-Bubble 0.7435 | DE-Selection 0.7319 | -0.0116 | [-0.0268, +0.0032] | 12-21-10 | 0.144 | 1.000 | -ns |

## DualEnd vs TD-Bubble baseline — DL20

| Model | Baseline | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | TD-Bubble 0.6264 | DE-Cocktail 0.6308 | +0.0044 | [-0.0164, +0.0274] | 21-28-5 | 0.698 | 1.000 | +ns |
| flan-t5-large | TD-Bubble 0.6264 | DE-Selection 0.6081 | -0.0183 | [-0.0481, +0.0122] | 20-32-2 | 0.246 | 1.000 | -ns |
| flan-t5-xl | TD-Bubble 0.6868 | DE-Cocktail 0.6795 | -0.0073 | [-0.0182, +0.0028] | 19-27-8 | 0.185 | 1.000 | -ns |
| flan-t5-xl | TD-Bubble 0.6868 | DE-Selection 0.6600 | -0.0268 | [-0.0433, -0.0113] | 14-34-6 | 0.001 | 0.023 | sig loss |
| flan-t5-xxl | TD-Bubble 0.6959 | DE-Cocktail 0.6895 | -0.0064 | [-0.0205, +0.0074] | 24-22-8 | 0.378 | 1.000 | -ns |
| flan-t5-xxl | TD-Bubble 0.6959 | DE-Selection 0.6754 | -0.0205 | [-0.0440, +0.0029] | 21-27-6 | 0.095 | 1.000 | -ns |
| qwen3-4b | TD-Bubble 0.6269 | DE-Cocktail 0.6454 | +0.0185 | [+0.0008, +0.0359] | 31-19-4 | 0.045 | 0.815 | +ns |
| qwen3-4b | TD-Bubble 0.6269 | DE-Selection 0.6627 | +0.0358 | [+0.0051, +0.0649] | 36-15-3 | 0.022 | 0.400 | +ns |
| qwen3-8b | TD-Bubble 0.6372 | DE-Cocktail 0.6678 | +0.0306 | [+0.0124, +0.0493] | 33-16-5 | 0.002 | 0.037 | sig win |
| qwen3-8b | TD-Bubble 0.6372 | DE-Selection 0.6583 | +0.0211 | [-0.0108, +0.0533] | 30-21-3 | 0.209 | 1.000 | +ns |
| qwen3-14b | TD-Bubble 0.7044 | DE-Cocktail 0.7051 | +0.0007 | [-0.0154, +0.0165] | 20-26-8 | 0.932 | 1.000 | +ns |
| qwen3-14b | TD-Bubble 0.7044 | DE-Selection 0.7003 | -0.0041 | [-0.0209, +0.0128] | 21-21-12 | 0.642 | 1.000 | -ns |
| qwen3.5-4b | TD-Bubble 0.6713 | DE-Cocktail 0.6768 | +0.0055 | [-0.0147, +0.0253] | 24-24-6 | 0.595 | 1.000 | +ns |
| qwen3.5-4b | TD-Bubble 0.6713 | DE-Selection 0.6487 | -0.0226 | [-0.0464, -0.0010] | 21-28-5 | 0.056 | 1.000 | -ns |
| qwen3.5-9b | TD-Bubble 0.6925 | DE-Cocktail 0.6984 | +0.0059 | [-0.0129, +0.0259] | 22-25-7 | 0.571 | 1.000 | +ns |
| qwen3.5-9b | TD-Bubble 0.6925 | DE-Selection 0.6927 | +0.0002 | [-0.0212, +0.0224] | 22-22-10 | 0.982 | 1.000 | +ns |
| qwen3.5-27b | TD-Bubble 0.7178 | DE-Cocktail 0.7186 | +0.0008 | [-0.0105, +0.0128] | 17-28-9 | 0.890 | 1.000 | +ns |
| qwen3.5-27b | TD-Bubble 0.7178 | DE-Selection 0.7122 | -0.0056 | [-0.0238, +0.0104] | 17-24-13 | 0.557 | 1.000 | -ns |

## DualEnd vs TD-Heap baseline — DL19

| Model | Baseline | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | TD-Heap 0.6541 | DE-Cocktail 0.6708 | +0.0167 | [-0.0045, +0.0383] | 24-14-5 | 0.138 | 1.000 | +ns |
| flan-t5-large | TD-Heap 0.6541 | DE-Selection 0.6420 | -0.0122 | [-0.0368, +0.0108] | 18-21-4 | 0.333 | 1.000 | -ns |
| flan-t5-xl | TD-Heap 0.6901 | DE-Cocktail 0.6884 | -0.0016 | [-0.0196, +0.0159] | 20-16-7 | 0.861 | 1.000 | -ns |
| flan-t5-xl | TD-Heap 0.6901 | DE-Selection 0.6792 | -0.0109 | [-0.0365, +0.0156] | 16-20-7 | 0.431 | 1.000 | -ns |
| flan-t5-xxl | TD-Heap 0.6846 | DE-Cocktail 0.7137 | +0.0291 | [+0.0068, +0.0547] | 22-14-7 | 0.014 | 0.251 | +ns |
| flan-t5-xxl | TD-Heap 0.6846 | DE-Selection 0.6974 | +0.0128 | [-0.0100, +0.0360] | 19-17-7 | 0.289 | 1.000 | +ns |
| qwen3-4b | TD-Heap 0.6775 | DE-Cocktail 0.6796 | +0.0022 | [-0.0375, +0.0412] | 24-16-3 | 0.915 | 1.000 | +ns |
| qwen3-4b | TD-Heap 0.6775 | DE-Selection 0.7220 | +0.0446 | [+0.0214, +0.0693] | 28-13-2 | <0.001 | 0.009 | sig win |
| qwen3-8b | TD-Heap 0.6819 | DE-Cocktail 0.7155 | +0.0336 | [+0.0037, +0.0641] | 26-12-5 | 0.035 | 0.637 | +ns |
| qwen3-8b | TD-Heap 0.6819 | DE-Selection 0.7158 | +0.0340 | [+0.0024, +0.0661] | 26-12-5 | 0.041 | 0.729 | +ns |
| qwen3-14b | TD-Heap 0.7447 | DE-Cocktail 0.7519 | +0.0071 | [-0.0113, +0.0287] | 19-15-9 | 0.533 | 1.000 | +ns |
| qwen3-14b | TD-Heap 0.7447 | DE-Selection 0.7475 | +0.0028 | [-0.0137, +0.0178] | 18-17-8 | 0.736 | 1.000 | +ns |
| qwen3.5-4b | TD-Heap 0.7087 | DE-Cocktail 0.7161 | +0.0074 | [-0.0147, +0.0289] | 18-19-6 | 0.516 | 1.000 | +ns |
| qwen3.5-4b | TD-Heap 0.7087 | DE-Selection 0.7022 | -0.0064 | [-0.0275, +0.0140] | 18-19-6 | 0.555 | 1.000 | -ns |
| qwen3.5-9b | TD-Heap 0.7329 | DE-Cocktail 0.7370 | +0.0042 | [-0.0125, +0.0207] | 23-13-7 | 0.632 | 1.000 | +ns |
| qwen3.5-9b | TD-Heap 0.7329 | DE-Selection 0.7309 | -0.0020 | [-0.0191, +0.0156] | 15-21-7 | 0.827 | 1.000 | -ns |
| qwen3.5-27b | TD-Heap 0.7449 | DE-Cocktail 0.7475 | +0.0026 | [-0.0161, +0.0213] | 15-18-10 | 0.792 | 1.000 | +ns |
| qwen3.5-27b | TD-Heap 0.7449 | DE-Selection 0.7319 | -0.0130 | [-0.0273, +0.0010] | 11-21-11 | 0.082 | 1.000 | -ns |

## DualEnd vs TD-Heap baseline — DL20

| Model | Baseline | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | TD-Heap 0.6100 | DE-Cocktail 0.6308 | +0.0208 | [-0.0029, +0.0426] | 34-17-3 | 0.081 | 1.000 | +ns |
| flan-t5-large | TD-Heap 0.6100 | DE-Selection 0.6081 | -0.0019 | [-0.0224, +0.0179] | 25-24-5 | 0.859 | 1.000 | -ns |
| flan-t5-xl | TD-Heap 0.6680 | DE-Cocktail 0.6795 | +0.0115 | [-0.0084, +0.0336] | 28-20-6 | 0.308 | 1.000 | +ns |
| flan-t5-xl | TD-Heap 0.6680 | DE-Selection 0.6600 | -0.0080 | [-0.0278, +0.0133] | 17-32-5 | 0.451 | 1.000 | -ns |
| flan-t5-xxl | TD-Heap 0.6790 | DE-Cocktail 0.6895 | +0.0104 | [-0.0088, +0.0313] | 24-23-7 | 0.324 | 1.000 | +ns |
| flan-t5-xxl | TD-Heap 0.6790 | DE-Selection 0.6754 | -0.0036 | [-0.0229, +0.0165] | 18-27-9 | 0.724 | 1.000 | -ns |
| qwen3-4b | TD-Heap 0.6488 | DE-Cocktail 0.6454 | -0.0034 | [-0.0316, +0.0263] | 23-29-2 | 0.820 | 1.000 | -ns |
| qwen3-4b | TD-Heap 0.6488 | DE-Selection 0.6627 | +0.0139 | [-0.0097, +0.0398] | 24-28-2 | 0.285 | 1.000 | +ns |
| qwen3-8b | TD-Heap 0.6532 | DE-Cocktail 0.6678 | +0.0146 | [-0.0118, +0.0399] | 30-19-5 | 0.285 | 1.000 | +ns |
| qwen3-8b | TD-Heap 0.6532 | DE-Selection 0.6583 | +0.0051 | [-0.0254, +0.0352] | 30-22-2 | 0.746 | 1.000 | +ns |
| qwen3-14b | TD-Heap 0.6962 | DE-Cocktail 0.7051 | +0.0090 | [-0.0152, +0.0342] | 25-20-9 | 0.491 | 1.000 | +ns |
| qwen3-14b | TD-Heap 0.6962 | DE-Selection 0.7003 | +0.0042 | [-0.0142, +0.0251] | 18-25-11 | 0.711 | 1.000 | +ns |
| qwen3.5-4b | TD-Heap 0.6521 | DE-Cocktail 0.6768 | +0.0247 | [+0.0000, +0.0500] | 28-22-4 | 0.058 | 1.000 | +ns |
| qwen3.5-4b | TD-Heap 0.6521 | DE-Selection 0.6487 | -0.0034 | [-0.0273, +0.0211] | 26-24-4 | 0.788 | 1.000 | -ns |
| qwen3.5-9b | TD-Heap 0.6950 | DE-Cocktail 0.6984 | +0.0034 | [-0.0139, +0.0214] | 23-25-6 | 0.720 | 1.000 | +ns |
| qwen3.5-9b | TD-Heap 0.6950 | DE-Selection 0.6927 | -0.0023 | [-0.0211, +0.0179] | 19-28-7 | 0.822 | 1.000 | -ns |
| qwen3.5-27b | TD-Heap 0.7104 | DE-Cocktail 0.7186 | +0.0082 | [-0.0049, +0.0229] | 24-21-9 | 0.266 | 1.000 | +ns |
| qwen3.5-27b | TD-Heap 0.7104 | DE-Selection 0.7122 | +0.0018 | [-0.0115, +0.0146] | 24-20-10 | 0.800 | 1.000 | +ns |

## Bidirectional vs TD-Bubble baseline — DL19

| Model | Baseline | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | TD-Bubble 0.6874 | BiDir-RRF 0.5820 | -0.1054 | [-0.1493, -0.0637] | 8-33-2 | <0.001 | <0.001 | sig loss |
| flan-t5-large | TD-Bubble 0.6874 | BiDir-Wt(a=0.7) 0.6147 | -0.0727 | [-0.1144, -0.0332] | 12-28-3 | <0.001 | 0.012 | sig loss |
| flan-t5-xl | TD-Bubble 0.6980 | BiDir-RRF 0.6845 | -0.0135 | [-0.0373, +0.0107] | 16-23-4 | 0.285 | 1.000 | -ns |
| flan-t5-xl | TD-Bubble 0.6980 | BiDir-Wt(a=0.7) 0.6810 | -0.0170 | [-0.0379, +0.0038] | 14-24-5 | 0.125 | 1.000 | -ns |
| flan-t5-xxl | TD-Bubble 0.7077 | BiDir-RRF 0.6905 | -0.0172 | [-0.0469, +0.0115] | 15-24-4 | 0.267 | 1.000 | -ns |
| flan-t5-xxl | TD-Bubble 0.7077 | BiDir-Wt(a=0.7) 0.6734 | -0.0343 | [-0.0610, -0.0092] | 14-26-3 | 0.012 | 0.219 | -ns |
| qwen3-4b | TD-Bubble 0.6491 | BiDir-RRF 0.6814 | +0.0323 | [+0.0005, +0.0686] | 20-19-4 | 0.070 | 1.000 | +ns |
| qwen3-4b | TD-Bubble 0.6491 | BiDir-Wt(a=0.7) 0.6608 | +0.0117 | [-0.0183, +0.0432] | 18-23-2 | 0.471 | 1.000 | +ns |
| qwen3-8b | TD-Bubble 0.6794 | BiDir-RRF 0.6826 | +0.0032 | [-0.0220, +0.0290] | 21-20-2 | 0.808 | 1.000 | +ns |
| qwen3-8b | TD-Bubble 0.6794 | BiDir-Wt(a=0.7) 0.6784 | -0.0010 | [-0.0246, +0.0223] | 22-18-3 | 0.934 | 1.000 | -ns |
| qwen3-14b | TD-Bubble 0.7455 | BiDir-RRF 0.7172 | -0.0283 | [-0.0578, -0.0010] | 16-20-7 | 0.057 | 1.000 | -ns |
| qwen3-14b | TD-Bubble 0.7455 | BiDir-Wt(a=0.7) 0.7200 | -0.0255 | [-0.0517, -0.0004] | 17-21-5 | 0.061 | 1.000 | -ns |
| qwen3.5-4b | TD-Bubble 0.7108 | BiDir-RRF 0.6614 | -0.0495 | [-0.0820, -0.0201] | 15-26-2 | 0.002 | 0.037 | sig loss |
| qwen3.5-4b | TD-Bubble 0.7108 | BiDir-Wt(a=0.7) 0.6714 | -0.0394 | [-0.0675, -0.0145] | 16-25-2 | 0.004 | 0.069 | -ns |
| qwen3.5-9b | TD-Bubble 0.7349 | BiDir-RRF 0.7101 | -0.0249 | [-0.0497, +0.0016] | 14-25-4 | 0.065 | 1.000 | -ns |
| qwen3.5-9b | TD-Bubble 0.7349 | BiDir-Wt(a=0.7) 0.7087 | -0.0262 | [-0.0497, -0.0019] | 12-26-5 | 0.042 | 0.753 | -ns |
| qwen3.5-27b | TD-Bubble 0.7435 | BiDir-RRF 0.7198 | -0.0237 | [-0.0461, -0.0005] | 10-24-9 | 0.052 | 0.939 | -ns |
| qwen3.5-27b | TD-Bubble 0.7435 | BiDir-Wt(a=0.7) 0.7229 | -0.0206 | [-0.0414, +0.0009] | 10-25-8 | 0.064 | 1.000 | -ns |

## Bidirectional vs TD-Bubble baseline — DL20

| Model | Baseline | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | TD-Bubble 0.6264 | BiDir-RRF 0.5580 | -0.0683 | [-0.1029, -0.0344] | 18-34-2 | <0.001 | 0.004 | sig loss |
| flan-t5-large | TD-Bubble 0.6264 | BiDir-Wt(a=0.7) 0.5805 | -0.0459 | [-0.0738, -0.0183] | 19-32-3 | 0.002 | 0.037 | sig loss |
| flan-t5-xl | TD-Bubble 0.6868 | BiDir-RRF 0.6509 | -0.0359 | [-0.0620, -0.0093] | 18-32-4 | 0.010 | 0.179 | -ns |
| flan-t5-xl | TD-Bubble 0.6868 | BiDir-Wt(a=0.7) 0.6528 | -0.0340 | [-0.0581, -0.0097] | 17-34-3 | 0.007 | 0.132 | -ns |
| flan-t5-xxl | TD-Bubble 0.6959 | BiDir-RRF 0.6673 | -0.0286 | [-0.0505, -0.0067] | 13-34-7 | 0.013 | 0.226 | -ns |
| flan-t5-xxl | TD-Bubble 0.6959 | BiDir-Wt(a=0.7) 0.6649 | -0.0310 | [-0.0528, -0.0098] | 12-37-5 | 0.005 | 0.087 | -ns |
| qwen3-4b | TD-Bubble 0.6269 | BiDir-RRF 0.6245 | -0.0024 | [-0.0295, +0.0248] | 24-28-2 | 0.868 | 1.000 | -ns |
| qwen3-4b | TD-Bubble 0.6269 | BiDir-Wt(a=0.7) 0.6376 | +0.0107 | [-0.0129, +0.0332] | 28-23-3 | 0.364 | 1.000 | +ns |
| qwen3-8b | TD-Bubble 0.6372 | BiDir-RRF 0.6600 | +0.0228 | [-0.0027, +0.0494] | 28-24-2 | 0.094 | 1.000 | +ns |
| qwen3-8b | TD-Bubble 0.6372 | BiDir-Wt(a=0.7) 0.6397 | +0.0025 | [-0.0276, +0.0301] | 23-28-3 | 0.874 | 1.000 | +ns |
| qwen3-14b | TD-Bubble 0.7044 | BiDir-RRF 0.6736 | -0.0308 | [-0.0504, -0.0117] | 17-36-1 | 0.003 | 0.049 | sig loss |
| qwen3-14b | TD-Bubble 0.7044 | BiDir-Wt(a=0.7) 0.6741 | -0.0303 | [-0.0522, -0.0105] | 17-34-3 | 0.005 | 0.092 | -ns |
| qwen3.5-4b | TD-Bubble 0.6713 | BiDir-RRF 0.6403 | -0.0310 | [-0.0610, -0.0029] | 17-35-2 | 0.039 | 0.699 | -ns |
| qwen3.5-4b | TD-Bubble 0.6713 | BiDir-Wt(a=0.7) 0.6567 | -0.0147 | [-0.0394, +0.0101] | 19-34-1 | 0.257 | 1.000 | -ns |
| qwen3.5-9b | TD-Bubble 0.6925 | BiDir-RRF 0.6647 | -0.0278 | [-0.0490, -0.0068] | 12-35-7 | 0.012 | 0.218 | -ns |
| qwen3.5-9b | TD-Bubble 0.6925 | BiDir-Wt(a=0.7) 0.6768 | -0.0157 | [-0.0366, +0.0048] | 19-27-8 | 0.148 | 1.000 | -ns |
| qwen3.5-27b | TD-Bubble 0.7178 | BiDir-RRF 0.6781 | -0.0397 | [-0.0591, -0.0207] | 13-35-6 | <0.001 | 0.004 | sig loss |
| qwen3.5-27b | TD-Bubble 0.7178 | BiDir-Wt(a=0.7) 0.6871 | -0.0307 | [-0.0481, -0.0139] | 12-35-7 | 0.001 | 0.019 | sig loss |

## Bidirectional vs TD-Heap baseline — DL19

| Model | Baseline | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | TD-Heap 0.6541 | BiDir-RRF 0.5820 | -0.0721 | [-0.1154, -0.0314] | 13-29-1 | 0.001 | 0.022 | sig loss |
| flan-t5-large | TD-Heap 0.6541 | BiDir-Wt(a=0.7) 0.6147 | -0.0394 | [-0.0756, -0.0049] | 14-25-4 | 0.039 | 0.700 | -ns |
| flan-t5-xl | TD-Heap 0.6901 | BiDir-RRF 0.6845 | -0.0056 | [-0.0301, +0.0203] | 14-24-5 | 0.677 | 1.000 | -ns |
| flan-t5-xl | TD-Heap 0.6901 | BiDir-Wt(a=0.7) 0.6810 | -0.0091 | [-0.0302, +0.0121] | 18-20-5 | 0.411 | 1.000 | -ns |
| flan-t5-xxl | TD-Heap 0.6846 | BiDir-RRF 0.6905 | +0.0059 | [-0.0221, +0.0331] | 18-19-6 | 0.680 | 1.000 | +ns |
| flan-t5-xxl | TD-Heap 0.6846 | BiDir-Wt(a=0.7) 0.6734 | -0.0111 | [-0.0363, +0.0129] | 17-20-6 | 0.398 | 1.000 | -ns |
| qwen3-4b | TD-Heap 0.6775 | BiDir-RRF 0.6814 | +0.0039 | [-0.0238, +0.0317] | 20-21-2 | 0.788 | 1.000 | +ns |
| qwen3-4b | TD-Heap 0.6775 | BiDir-Wt(a=0.7) 0.6608 | -0.0167 | [-0.0444, +0.0099] | 18-22-3 | 0.241 | 1.000 | -ns |
| qwen3-8b | TD-Heap 0.6819 | BiDir-RRF 0.6826 | +0.0008 | [-0.0283, +0.0319] | 21-19-3 | 0.962 | 1.000 | +ns |
| qwen3-8b | TD-Heap 0.6819 | BiDir-Wt(a=0.7) 0.6784 | -0.0034 | [-0.0323, +0.0250] | 21-20-2 | 0.816 | 1.000 | -ns |
| qwen3-14b | TD-Heap 0.7447 | BiDir-RRF 0.7172 | -0.0276 | [-0.0558, -0.0007] | 15-23-5 | 0.060 | 1.000 | -ns |
| qwen3-14b | TD-Heap 0.7447 | BiDir-Wt(a=0.7) 0.7200 | -0.0247 | [-0.0482, -0.0012] | 12-25-6 | 0.048 | 0.856 | -ns |
| qwen3.5-4b | TD-Heap 0.7087 | BiDir-RRF 0.6614 | -0.0473 | [-0.0810, -0.0158] | 15-24-4 | 0.005 | 0.098 | -ns |
| qwen3.5-4b | TD-Heap 0.7087 | BiDir-Wt(a=0.7) 0.6714 | -0.0372 | [-0.0650, -0.0115] | 18-22-3 | 0.008 | 0.145 | -ns |
| qwen3.5-9b | TD-Heap 0.7329 | BiDir-RRF 0.7101 | -0.0228 | [-0.0439, -0.0020] | 16-22-5 | 0.042 | 0.750 | -ns |
| qwen3.5-9b | TD-Heap 0.7329 | BiDir-Wt(a=0.7) 0.7087 | -0.0241 | [-0.0432, -0.0056] | 16-22-5 | 0.017 | 0.309 | -ns |
| qwen3.5-27b | TD-Heap 0.7449 | BiDir-RRF 0.7198 | -0.0251 | [-0.0504, -0.0007] | 12-24-7 | 0.057 | 1.000 | -ns |
| qwen3.5-27b | TD-Heap 0.7449 | BiDir-Wt(a=0.7) 0.7229 | -0.0220 | [-0.0442, -0.0015] | 12-18-13 | 0.051 | 0.913 | -ns |

## Bidirectional vs TD-Heap baseline — DL20

| Model | Baseline | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | TD-Heap 0.6100 | BiDir-RRF 0.5580 | -0.0519 | [-0.1005, -0.0066] | 20-32-2 | 0.034 | 0.618 | -ns |
| flan-t5-large | TD-Heap 0.6100 | BiDir-Wt(a=0.7) 0.5805 | -0.0295 | [-0.0680, +0.0056] | 23-28-3 | 0.128 | 1.000 | -ns |
| flan-t5-xl | TD-Heap 0.6680 | BiDir-RRF 0.6509 | -0.0171 | [-0.0405, +0.0070] | 20-28-6 | 0.177 | 1.000 | -ns |
| flan-t5-xl | TD-Heap 0.6680 | BiDir-Wt(a=0.7) 0.6528 | -0.0152 | [-0.0352, +0.0049] | 21-27-6 | 0.146 | 1.000 | -ns |
| flan-t5-xxl | TD-Heap 0.6790 | BiDir-RRF 0.6673 | -0.0118 | [-0.0350, +0.0133] | 18-31-5 | 0.344 | 1.000 | -ns |
| flan-t5-xxl | TD-Heap 0.6790 | BiDir-Wt(a=0.7) 0.6649 | -0.0141 | [-0.0335, +0.0059] | 16-33-5 | 0.169 | 1.000 | -ns |
| qwen3-4b | TD-Heap 0.6488 | BiDir-RRF 0.6245 | -0.0243 | [-0.0484, +0.0002] | 22-30-2 | 0.056 | 1.000 | -ns |
| qwen3-4b | TD-Heap 0.6488 | BiDir-Wt(a=0.7) 0.6376 | -0.0112 | [-0.0328, +0.0120] | 18-33-3 | 0.340 | 1.000 | -ns |
| qwen3-8b | TD-Heap 0.6532 | BiDir-RRF 0.6600 | +0.0068 | [-0.0178, +0.0328] | 23-27-4 | 0.607 | 1.000 | +ns |
| qwen3-8b | TD-Heap 0.6532 | BiDir-Wt(a=0.7) 0.6397 | -0.0135 | [-0.0412, +0.0129] | 23-28-3 | 0.341 | 1.000 | -ns |
| qwen3-14b | TD-Heap 0.6962 | BiDir-RRF 0.6736 | -0.0225 | [-0.0432, -0.0016] | 20-32-2 | 0.039 | 0.707 | -ns |
| qwen3-14b | TD-Heap 0.6962 | BiDir-Wt(a=0.7) 0.6741 | -0.0220 | [-0.0440, -0.0014] | 19-32-3 | 0.050 | 0.894 | -ns |
| qwen3.5-4b | TD-Heap 0.6521 | BiDir-RRF 0.6403 | -0.0118 | [-0.0319, +0.0088] | 22-30-2 | 0.263 | 1.000 | -ns |
| qwen3.5-4b | TD-Heap 0.6521 | BiDir-Wt(a=0.7) 0.6567 | +0.0045 | [-0.0122, +0.0213] | 28-24-2 | 0.601 | 1.000 | +ns |
| qwen3.5-9b | TD-Heap 0.6950 | BiDir-RRF 0.6647 | -0.0303 | [-0.0481, -0.0130] | 14-36-4 | 0.002 | 0.030 | sig loss |
| qwen3.5-9b | TD-Heap 0.6950 | BiDir-Wt(a=0.7) 0.6768 | -0.0182 | [-0.0345, -0.0020] | 15-32-7 | 0.033 | 0.594 | -ns |
| qwen3.5-27b | TD-Heap 0.7104 | BiDir-RRF 0.6781 | -0.0323 | [-0.0522, -0.0131] | 14-35-5 | 0.002 | 0.037 | sig loss |
| qwen3.5-27b | TD-Heap 0.7104 | BiDir-Wt(a=0.7) 0.6871 | -0.0234 | [-0.0405, -0.0074] | 13-32-9 | 0.007 | 0.120 | -ns |

