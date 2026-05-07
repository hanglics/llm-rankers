# Significance Tests

## Method
- Per-query `ndcg_cut_10` values are read from the saved `.eval` files under `results/`.
- Each comparison uses the best method within a challenger family against the best TopDown baseline for that model-dataset configuration.
- Statistical test: two-sided paired approximate randomization with `100,000` samples.
- Uncertainty: paired bootstrap 95% CI on mean delta with `20,000` resamples.
- Multiple testing: Bonferroni correction within each family across the 18 configs.

## Summary

| Family | Mean delta vs best TopDown | Positive deltas | Bonferroni-significant wins | Bonferroni-significant losses |
|---|---:|---:|---:|---:|
| DualEnd | +0.0058 | 14/18 | 1 | 0 |
| BottomUp | -0.0616 | 0/18 | 0 | 6 |
| BiDir | -0.0232 | 3/18 | 0 | 3 |

## DualEnd vs Best TopDown

| Model | Dataset | TopDown | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | DL19 | TD-Bubble 0.6874 | DE-Cocktail 0.6708 | -0.0165 | [-0.0397, +0.0042] | 14-21-8 | 0.159 | 1.000 | negative, ns |
| flan-t5-large | DL20 | TD-Bubble 0.6264 | DE-Cocktail 0.6308 | +0.0044 | [-0.0162, +0.0268] | 21-28-5 | 0.697 | 1.000 | positive, ns |
| flan-t5-xl | DL19 | TD-Bubble 0.6980 | DE-Cocktail 0.6884 | -0.0096 | [-0.0259, +0.0048] | 18-16-9 | 0.246 | 1.000 | negative, ns |
| flan-t5-xl | DL20 | TD-Bubble 0.6868 | DE-Cocktail 0.6795 | -0.0073 | [-0.0181, +0.0028] | 19-27-8 | 0.184 | 1.000 | negative, ns |
| flan-t5-xxl | DL19 | TD-Bubble 0.7077 | DE-Cocktail 0.7137 | +0.0060 | [-0.0081, +0.0208] | 17-18-8 | 0.434 | 1.000 | positive, ns |
| flan-t5-xxl | DL20 | TD-Bubble 0.6959 | DE-Cocktail 0.6895 | -0.0064 | [-0.0207, +0.0075] | 24-22-8 | 0.381 | 1.000 | negative, ns |
| qwen3-14b | DL19 | TD-Bubble 0.7455 | DE-Cocktail 0.7519 | +0.0064 | [-0.0056, +0.0190] | 17-17-9 | 0.329 | 1.000 | positive, ns |
| qwen3-14b | DL20 | TD-Bubble 0.7044 | DE-Cocktail 0.7051 | +0.0007 | [-0.0150, +0.0166] | 20-26-8 | 0.933 | 1.000 | positive, ns |
| qwen3-4b | DL19 | TD-Heap 0.6775 | DE-Selection 0.7220 | +0.0446 | [+0.0216, +0.0695] | 28-13-2 | <0.001 | 0.010 | significant win |
| qwen3-4b | DL20 | TD-Heap 0.6488 | DE-Selection 0.6627 | +0.0139 | [-0.0100, +0.0395] | 24-28-2 | 0.284 | 1.000 | positive, ns |
| qwen3-8b | DL19 | TD-Heap 0.6819 | DE-Selection 0.7158 | +0.0340 | [+0.0033, +0.0654] | 26-12-5 | 0.040 | 0.727 | positive, ns |
| qwen3-8b | DL20 | TD-Heap 0.6532 | DE-Cocktail 0.6678 | +0.0146 | [-0.0122, +0.0400] | 30-19-5 | 0.285 | 1.000 | positive, ns |
| qwen3.5-27b | DL19 | TD-Heap 0.7449 | DE-Cocktail 0.7475 | +0.0026 | [-0.0165, +0.0211] | 15-18-10 | 0.791 | 1.000 | positive, ns |
| qwen3.5-27b | DL20 | TD-Bubble 0.7178 | DE-Cocktail 0.7186 | +0.0008 | [-0.0104, +0.0128] | 17-28-9 | 0.892 | 1.000 | positive, ns |
| qwen3.5-4b | DL19 | TD-Bubble 0.7108 | DE-Cocktail 0.7161 | +0.0052 | [-0.0155, +0.0253] | 20-19-4 | 0.620 | 1.000 | positive, ns |
| qwen3.5-4b | DL20 | TD-Bubble 0.6713 | DE-Cocktail 0.6768 | +0.0055 | [-0.0148, +0.0252] | 24-24-6 | 0.594 | 1.000 | positive, ns |
| qwen3.5-9b | DL19 | TD-Bubble 0.7349 | DE-Cocktail 0.7370 | +0.0021 | [-0.0192, +0.0256] | 20-19-4 | 0.862 | 1.000 | positive, ns |
| qwen3.5-9b | DL20 | TD-Heap 0.6950 | DE-Cocktail 0.6984 | +0.0034 | [-0.0143, +0.0218] | 23-25-6 | 0.721 | 1.000 | positive, ns |

## BottomUp vs Best TopDown

| Model | Dataset | TopDown | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | DL19 | TD-Bubble 0.6874 | BU-Bubble 0.4571 | -0.2302 | [-0.2922, -0.1725] | 6-34-3 | <0.001 | <0.001 | significant loss |
| flan-t5-large | DL20 | TD-Bubble 0.6264 | BU-Bubble 0.4116 | -0.2148 | [-0.2832, -0.1477] | 7-45-2 | <0.001 | <0.001 | significant loss |
| flan-t5-xl | DL19 | TD-Bubble 0.6980 | BU-Bubble 0.6730 | -0.0250 | [-0.0517, +0.0014] | 13-21-9 | 0.077 | 1.000 | negative, ns |
| flan-t5-xl | DL20 | TD-Bubble 0.6868 | BU-Bubble 0.6691 | -0.0177 | [-0.0445, +0.0074] | 23-27-4 | 0.194 | 1.000 | negative, ns |
| flan-t5-xxl | DL19 | TD-Bubble 0.7077 | BU-Bubble 0.6936 | -0.0141 | [-0.0345, +0.0059] | 15-24-4 | 0.185 | 1.000 | negative, ns |
| flan-t5-xxl | DL20 | TD-Bubble 0.6959 | BU-Bubble 0.6811 | -0.0148 | [-0.0404, +0.0102] | 20-29-5 | 0.268 | 1.000 | negative, ns |
| qwen3-14b | DL19 | TD-Bubble 0.7455 | BU-Heap 0.6966 | -0.0489 | [-0.0926, -0.0112] | 13-23-7 | 0.022 | 0.394 | negative, ns |
| qwen3-14b | DL20 | TD-Bubble 0.7044 | BU-Bubble 0.6395 | -0.0649 | [-0.0964, -0.0359] | 13-39-2 | <0.001 | 0.001 | significant loss |
| qwen3-4b | DL19 | TD-Heap 0.6775 | BU-Bubble 0.6305 | -0.0470 | [-0.0879, -0.0095] | 17-25-1 | 0.024 | 0.430 | negative, ns |
| qwen3-4b | DL20 | TD-Heap 0.6488 | BU-Heap 0.5963 | -0.0525 | [-0.0939, -0.0100] | 16-38-0 | 0.019 | 0.335 | negative, ns |
| qwen3-8b | DL19 | TD-Heap 0.6819 | BU-Heap 0.6431 | -0.0388 | [-0.0803, +0.0011] | 14-24-5 | 0.069 | 1.000 | negative, ns |
| qwen3-8b | DL20 | TD-Heap 0.6532 | BU-Heap 0.5963 | -0.0569 | [-0.0923, -0.0213] | 17-36-1 | 0.003 | 0.059 | negative, ns |
| qwen3.5-27b | DL19 | TD-Heap 0.7449 | BU-Bubble 0.7336 | -0.0113 | [-0.0445, +0.0174] | 15-19-9 | 0.525 | 1.000 | negative, ns |
| qwen3.5-27b | DL20 | TD-Bubble 0.7178 | BU-Bubble 0.7004 | -0.0174 | [-0.0415, +0.0064] | 18-28-8 | 0.166 | 1.000 | negative, ns |
| qwen3.5-4b | DL19 | TD-Bubble 0.7108 | BU-Heap 0.6158 | -0.0950 | [-0.1448, -0.0480] | 12-29-2 | <0.001 | 0.005 | significant loss |
| qwen3.5-4b | DL20 | TD-Bubble 0.6713 | BU-Bubble 0.5963 | -0.0749 | [-0.1164, -0.0325] | 12-37-5 | <0.001 | 0.016 | significant loss |
| qwen3.5-9b | DL19 | TD-Bubble 0.7349 | BU-Heap 0.6779 | -0.0570 | [-0.0905, -0.0244] | 12-27-4 | 0.001 | 0.025 | significant loss |
| qwen3.5-9b | DL20 | TD-Heap 0.6950 | BU-Bubble 0.6677 | -0.0272 | [-0.0558, +0.0030] | 16-35-3 | 0.074 | 1.000 | negative, ns |

## BiDir vs Best TopDown

| Model | Dataset | TopDown | Challenger | Delta | 95% CI | Wins-Losses-Ties | Raw p | Bonferroni p | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| flan-t5-large | DL19 | TD-Bubble 0.6874 | BiDir-Wt(a=0.7) 0.6147 | -0.0727 | [-0.1140, -0.0332] | 12-28-3 | <0.001 | 0.013 | significant loss |
| flan-t5-large | DL20 | TD-Bubble 0.6264 | BiDir-Wt(a=0.7) 0.5805 | -0.0459 | [-0.0741, -0.0181] | 19-32-3 | 0.002 | 0.035 | significant loss |
| flan-t5-xl | DL19 | TD-Bubble 0.6980 | BiDir-RRF 0.6845 | -0.0135 | [-0.0367, +0.0103] | 16-23-4 | 0.285 | 1.000 | negative, ns |
| flan-t5-xl | DL20 | TD-Bubble 0.6868 | BiDir-Wt(a=0.7) 0.6528 | -0.0340 | [-0.0575, -0.0097] | 17-34-3 | 0.008 | 0.143 | negative, ns |
| flan-t5-xxl | DL19 | TD-Bubble 0.7077 | BiDir-RRF 0.6905 | -0.0172 | [-0.0466, +0.0116] | 15-24-4 | 0.266 | 1.000 | negative, ns |
| flan-t5-xxl | DL20 | TD-Bubble 0.6959 | BiDir-RRF 0.6673 | -0.0286 | [-0.0504, -0.0068] | 13-34-7 | 0.013 | 0.226 | negative, ns |
| qwen3-14b | DL19 | TD-Bubble 0.7455 | BiDir-Wt(a=0.7) 0.7200 | -0.0255 | [-0.0516, +0.0000] | 17-21-5 | 0.060 | 1.000 | negative, ns |
| qwen3-14b | DL20 | TD-Bubble 0.7044 | BiDir-Wt(a=0.7) 0.6741 | -0.0303 | [-0.0520, -0.0101] | 17-34-3 | 0.005 | 0.097 | negative, ns |
| qwen3-4b | DL19 | TD-Heap 0.6775 | BiDir-RRF 0.6814 | +0.0039 | [-0.0235, +0.0318] | 20-21-2 | 0.788 | 1.000 | positive, ns |
| qwen3-4b | DL20 | TD-Heap 0.6488 | BiDir-Wt(a=0.7) 0.6376 | -0.0112 | [-0.0328, +0.0118] | 18-33-3 | 0.338 | 1.000 | negative, ns |
| qwen3-8b | DL19 | TD-Heap 0.6819 | BiDir-RRF 0.6826 | +0.0008 | [-0.0283, +0.0314] | 21-19-3 | 0.961 | 1.000 | positive, ns |
| qwen3-8b | DL20 | TD-Heap 0.6532 | BiDir-RRF 0.6600 | +0.0068 | [-0.0178, +0.0328] | 23-27-4 | 0.606 | 1.000 | positive, ns |
| qwen3.5-27b | DL19 | TD-Heap 0.7449 | BiDir-Wt(a=0.7) 0.7229 | -0.0220 | [-0.0440, -0.0015] | 12-18-13 | 0.051 | 0.917 | negative, ns |
| qwen3.5-27b | DL20 | TD-Bubble 0.7178 | BiDir-Wt(a=0.7) 0.6871 | -0.0307 | [-0.0479, -0.0138] | 12-35-7 | <0.001 | 0.016 | significant loss |
| qwen3.5-4b | DL19 | TD-Bubble 0.7108 | BiDir-Wt(a=0.7) 0.6714 | -0.0394 | [-0.0670, -0.0145] | 16-25-2 | 0.004 | 0.072 | negative, ns |
| qwen3.5-4b | DL20 | TD-Bubble 0.6713 | BiDir-Wt(a=0.7) 0.6567 | -0.0147 | [-0.0390, +0.0105] | 19-34-1 | 0.256 | 1.000 | negative, ns |
| qwen3.5-9b | DL19 | TD-Bubble 0.7349 | BiDir-RRF 0.7101 | -0.0249 | [-0.0502, +0.0013] | 14-25-4 | 0.065 | 1.000 | negative, ns |
| qwen3.5-9b | DL20 | TD-Heap 0.6950 | BiDir-Wt(a=0.7) 0.6768 | -0.0182 | [-0.0344, -0.0023] | 15-32-7 | 0.033 | 0.592 | negative, ns |

