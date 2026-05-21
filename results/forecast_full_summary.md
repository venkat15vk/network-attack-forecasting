# Full-dataset experimental result

Run on the Loghub OpenSSH corpus (28 days, 542,861 events, 1,241 source IPs)
using `src/run_forecast_fast.py --lookback 5 --horizon 15 --bucket 1`.

- Cells generated: 18,590 (from 52.8M anchor-minute candidates)
- Target rate (any attack-related event in next 15 min): 26.3%
- Target count: mean 23.2, median 0, max 1,041, std 72.9
- Wall time: 57.6 s

## Classification (any attack-related event in next horizon)

| Method        | AUC-PR              | AUC-ROC             |
| ------------- | ------------------- | ------------------- |
| **RF[all]**   | **0.952 ± 0.022**   | **0.970 ± 0.008**   |
| RF[agg]       | 0.927 ± 0.035       | 0.948 ± 0.023       |
| persist[>0]   | 0.871 ± 0.039       | 0.919 ± 0.009       |
| RF[outlier]   | 0.742 ± 0.124       | 0.880 ± 0.049       |
| mean_baseline | 0.409 ± 0.115       | 0.500 ± 0.000       |

Paired t-tests (RF[all] vs each):

- vs RF[agg]:        Δ=+0.025, wins=5/5, p=0.029
- vs persist[>0]:    Δ=+0.081, wins=5/5, p=0.0005
- vs RF[outlier]:    Δ=+0.209, wins=5/5, p=0.011
- vs mean_baseline:  Δ=+0.543, wins=5/5, p=0.0002

## Regression (count of attack-related events in next horizon)

| Method        | MAE             | RMSE              | R²                  |
| ------------- | --------------- | ----------------- | ------------------- |
| RF[all]       | 28.9 ± 8.8      | 72.8 ± 14.2       | +0.244 ± 0.43       |
| persist[×3]   | 27.8 ± 4.8      | 92.8 ± 8.2        | -0.120 ± 0.09       |
| RF[agg]       | 35.6 ± 25.6     | 101.1 ± 51.1      | -0.869 ± 2.47       |
| mean_baseline | 39.1 ± 4.0      | 88.6 ± 11.2       | -0.013 ± 0.02       |
| RF[outlier]   | 43.7 ± 4.0      | 94.9 ± 9.0        | -0.182 ± 0.20       |
