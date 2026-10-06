# Final AGE-6 and Age Model Results

Source: frozen `age6_formal_trials.csv` (12 formal rows only). Figures and AGE-6 statistics use only this file.

## A. AgeDB FP32 vs INT8

| Metric | FP32 | INT8 | Notes |
|---|---:|---:|---|
| Test MAE | 9.6941 years | 9.7239 years | INT8 minus FP32: +0.0297 years (+0.31%) |
| Absolute-error STD | 7.6447 | 7.7231 | years |
| Median absolute error | 8.0905 | 8.0479 | years |
| RMSE | 12.3458 | 12.4177 | years |
| Mean signed error | +0.9571 | +0.0548 | years |

| Optimization metric | FP32 | INT8 | Change |
|---|---:|---:|---:|
| Mean backbone latency | 33.9684 ms | 8.3335 ms | 4.0761× speedup; 75.47% reduction |
| Model size | 89.67 MiB | 23.17 MiB | 74.16% reduction; 3.8693× ratio |

Latency is **backbone invoke-only, batch size 1**; it excludes the full web application.

## B. AGE-6 Base Age Consistency

| N | Mean signed visual age difference | Mean absolute visual age difference | Median absolute visual age difference | Signed difference STD | Pearson r | Spearman ρ |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | -1.514 years | 9.508 years | 11.831 years | 11.415 years | 0.963 | 0.946 |

Visual age difference is ML Estimated Age minus witness age; it is not called prediction error.

## C. AGE-6 Refinement Direction

| Condition | Success | Mean age change | Expected direction |
|---|---:|---:|---|
| Younger | 5/6 (83.33%) | -8.111 years | < 0 |
| Older | 5/6 (83.33%) | +9.862 years | > 0 |
| Overall | 10/12 (83.33%) | — | — |
| Mean absolute change | — | 9.555 years | — |
| Median absolute change | — | 10.153 years | — |

## D. Results by Witness Age

| Witness age | N | Mean base ML age | Mean signed visual age difference | Mean absolute visual age difference | Base ML age sample STD |
|---:|---:|---:|---:|---:|---:|
| 20 | 4 | 31.843 | +11.843 | 11.843 | 1.129 |
| 40 | 4 | 38.276 | -1.724 | 2.019 | 2.369 |
| 60 | 4 | 45.340 | -14.660 | 14.660 | 1.582 |

## E. All 12 Formal Trials

| ID | Condition | Witness age | Base ML age | Refined ML age | Change | Visual age difference | Direction success |
|---|---|---:|---:|---:|---:|---:|---|
| AGE20-01 | Younger | 20 | 32.567 | 23.951 | -8.616 | +12.567 | Yes |
| AGE20-02 | Younger | 20 | 31.095 | 28.997 | -2.098 | +11.095 | Yes |
| AGE20-03 | Older | 20 | 30.686 | 32.848 | +2.163 | +10.686 | Yes |
| AGE20-04 | Older | 20 | 33.025 | 29.909 | -3.117 | +13.025 | No |
| AGE40-01 | Younger | 40 | 35.422 | 35.718 | +0.296 | -4.578 | No |
| AGE40-02 | Younger | 40 | 37.282 | 30.719 | -6.562 | -2.718 | Yes |
| AGE40-03 | Older | 40 | 39.809 | 53.704 | +13.895 | -0.191 | Yes |
| AGE40-04 | Older | 40 | 40.590 | 52.280 | +11.690 | +0.590 | Yes |
| AGE60-01 | Younger | 60 | 45.157 | 28.985 | -16.173 | -14.843 | Yes |
| AGE60-02 | Younger | 60 | 45.747 | 30.235 | -15.513 | -14.253 | Yes |
| AGE60-03 | Older | 60 | 43.319 | 65.577 | +22.257 | -16.681 | Yes |
| AGE60-04 | Older | 60 | 47.135 | 59.418 | +12.283 | -12.865 | Yes |
