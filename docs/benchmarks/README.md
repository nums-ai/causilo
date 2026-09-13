# Benchmarks

Performance comparisons use Causilo results alongside the baseline result files provided by TabArena. Our local reruns of TabICLv2 and TabPFN-3 are used only for the resource comparison below.

## Resource comparison

All three models were rerun locally on the same H100 80 GB server, using one GPU and eight physical CPU cores per job. Times are median seconds per 1,000 rows; memory is mean peak usage during fit only.

| Task           | Model    |   Fit (s/1k) |   Predict (s/1k) |   CPU (GiB) |   GPU (GiB) |
|:---------------|:---------|-------------:|-----------------:|------------:|------------:|
| Overall | **Causilo** | **2.504** | **0.251** | **1.94** | 8.15 |
| Overall | TabICLv2 | 3.449 | 0.303 | 2 | 8.37 |
| Overall | TabPFN-3 | 4.18 | 0.686 | 2.87 | **0.88** |
| Classification | **Causilo** | **2.568** | **0.272** | **1.95** | 8.36 |
| Classification | TabICLv2 | 3.51 | 0.335 | 2 | 8.96 |
| Classification | TabPFN-3 | 4.185 | 0.703 | 2.9 | **0.81** |
| Regression | **Causilo** | **1.682** | **0.212** | **1.92** | 7.54 |
| Regression | TabICLv2 | 1.909 | 0.216 | 1.99 | 6.64 |
| Regression | TabPFN-3 | 3.19 | 0.498 | 2.81 | **1.07** |

[Evaluation provenance and software versions](provenance.json).
