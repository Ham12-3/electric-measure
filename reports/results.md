## Results (forecast weather)

| Horizon | Model | MAE (MW) | RMSE (MW) | MAPE | Skill vs baseline | Pinball q10 / q90 (MW) | 80% interval coverage |
|---|---|---|---|---|---|---|---|
| 1h | Same hour last week | 2,057 | 2,781 | 8.17% | 0% | – | – |
| 1h | Prophet | 1,447 | 1,823 | 5.98% | 30% | – | – |
| 1h | **XGBoost** | 343 | 453 | 1.39% | 83% | – | – |
| 1h | TFT | 1,032 | 1,329 | 4.25% | 50% | 231 / 254 | 68% |
| 24h | Same hour last week | 2,053 | 2,775 | 8.15% | 0% | – | – |
| 24h | Prophet | 1,472 | 1,856 | 6.08% | 28% | – | – |
| 24h | **XGBoost** | 991 | 1,348 | 4.10% | 52% | – | – |
| 24h | TFT | 1,021 | 1,330 | 4.22% | 50% | 246 / 240 | 64% |
| 168h | Same hour last week | 2,053 | 2,777 | 8.15% | 0% | – | – |
| 168h | Prophet | 1,844 | 2,392 | 7.52% | 10% | – | – |
| 168h | **XGBoost** | 1,405 | 1,907 | 5.76% | 32% | – | – |
| 168h | TFT | 1,510 | 2,049 | 6.04% | 26% | 391 / 472 | 49% |


## XGBoost: actual vs forecast weather (MAPE %)

|   horizon |   forecast |   actual |   inflation_% |
|----------:|-----------:|---------:|--------------:|
|         1 |       1.39 |     1.39 |         -0.06 |
|        24 |       4.1  |     3.99 |          2.68 |
|       168 |       5.76 |     4.6  |         20.11 |


## XGBoost top features (% of gain)

| feature             |    1 |    24 |   168 |
|:--------------------|-----:|------:|------:|
| lag_1               | 77   | nan   | nan   |
| lag_336             |  7.2 |  10.9 |  30.7 |
| hour                |  3.9 |   1.1 |   6.1 |
| lag_2               |  3.3 | nan   | nan   |
| lag_24              |  1.9 |  35.2 | nan   |
| lag_168             |  1.2 |  28.8 |  31.2 |
| is_holiday_ew       |  0.8 |   2.6 |   3.3 |
| shortwave_radiation |  0.7 |   1.2 |   1.9 |
| lag_48              |  0.6 |   4.5 | nan   |
| is_christmas_period |  0.6 |   1.8 |   6.1 |
| is_weekend          |  0.6 |   5.4 |   4.8 |
| lag_3               |  0.5 | nan   | nan   |
