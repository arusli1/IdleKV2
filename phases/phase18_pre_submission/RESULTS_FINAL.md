# Phase 18 RESULTS FINAL

Aggregated from queued reruns. NO PAPER EDITS APPLIED.


---

### K-sweep (Qwen, n=12 × 3 partitions = 36 obs/K)

| K | A | B_match | RepairKV | Refresh-K | Refresh-K-budgeted | PageSummary-Quest-inspired | RepairKV-no-burst | Oracle-K |
|---|---|---|---|---|---|---|---|---|
| 32 | 1.000 | 0.208 | 0.375 | 1.000 | 1.000 | 0.208 | 0.500 | 0.861 |
| 64 | 1.000 | 0.208 | 0.639 | 1.000 | 1.000 | 0.208 | 0.569 | 1.000 |
| 80 | 1.000 | 0.194 | 0.778 | 1.000 | 1.000 | 0.208 | 0.569 | 1.000 |
| 96 | 1.000 | 0.208 | 0.917 | 1.000 | 1.000 | 0.194 | 0.653 | 1.000 |
| 128 | 1.000 | 0.181 | 1.000 | 1.000 | 0.986 | 0.194 | 0.736 | 1.000 |


---

### Tight-budget multiplier sweep (Qwen, K=96, n=36)

| mult | budget (ms) | RepairKV | Refresh-K-budgeted | Δ vs RKB | PageSummary | Δ vs PSum | RKB cap fires | RKB positions scored |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 347 | 0.917 | 0.389 | +0.528 | 0.194 | +0.722 | 36/36 | 8107/32768 |
| 0.10 | 708 | 0.917 | 0.667 | +0.250 | 0.194 | +0.722 | 36/36 | 15986/32768 |
| 0.30 | 2110 | 0.917 | 1.000 | -0.083 | 0.194 | +0.722 | 0/36 | 32768/32768 |
| 1.05 | 7114 | 0.917 | 1.000 | -0.083 | 0.306 | +0.611 | 0/36 | 32768/32768 |


---



---

### Tight K-sweep at 150 ms absolute budget (Qwen, 4Q, n=36/K)

| K | A | B_match | RepairKV | Refresh-K | Refresh-K-budgeted | PageSummary-Quest-inspired | RepairKV-no-burst |
|---|---|---|---|---|---|---|---|
| 32 | 1.000 | 0.208 | 0.375 | 1.000 | 0.500 | 0.208 | 0.500 |
| 64 | 1.000 | 0.208 | 0.639 | 1.000 | 0.486 | 0.208 | 0.569 |
| 96 | 1.000 | 0.208 | 0.917 | 1.000 | 0.472 | 0.194 | 0.653 |
| 128 | 1.000 | 0.181 | 1.000 | 1.000 | 0.514 | 0.194 | 0.736 |