**Reward separability — AUROC for distinguishing SOLVED vs FAILED AND nodes:**

| component | action | n_solved | n_failed | AUROC | Δ mean |
|---|---|---:|---:|---:|---:|
| r_env | all | 1562 | 25297 | 0.981 | +0.360 |
| r_env | tactic | 1399 | 21855 | 0.989 | +0.366 |
| r_env | skeleton | 163 | 3442 | 0.932 | +0.324 |
| r_dep | all | 1562 | 25297 | 0.954 | +0.893 |
| r_dep | tactic | 1399 | 21855 | 0.966 | +0.914 |
| r_dep | skeleton | 163 | 3442 | 0.849 | +0.696 |
| Q | all | 1562 | 25297 | 0.998 | +1.468 |
| Q | tactic | 1399 | 21855 | 0.998 | +1.280 |
| Q | skeleton | 163 | 3442 | 1.000 | +3.082 |

**Per-problem root-Q vs eventual solve:**

| metric | value |
|---|---:|
| n problems | 277 |
| AUROC (max-root-Q vs solve) | 0.966 |
| mean root-Q (solved) | 2.249 |
| mean root-Q (failed) | 0.960 |