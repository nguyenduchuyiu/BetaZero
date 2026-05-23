**Search cost (GammaZero):**

| metric | mean | median | std | max | sum |
|---|---:|---:|---:|---:|---:|
| n_nodes | 106.63 | 15.00 | 161.03 | 568 | 29536 |
| n_or | 8.49 | 1.00 | 12.72 | 60 | 2352 |
| n_and | 98.14 | 14.00 | 149.40 | 512 | 27184 |
| n_tactic_and | 83.95 | 14.00 | 127.61 | 464 | 23254 |
| n_skeleton_and | 14.19 | 0.00 | 25.44 | 133 | 3930 |
| used_total | 98.87 | 14.00 | 150.75 | 512 | 27388 |
| lean_verify_calls | 98.87 | 14.00 | 150.75 | 512 | 27388 |
| patch_verify_calls | 81.12 | 13.00 | 121.80 | 459 | 22469 |

**Solved vs failed problems:**

| metric | solved (mean) | failed (mean) |
|---|---:|---:|
| n_nodes | 108.51 | 65.00 |
| lean_verify_calls | 100.45 | 64.00 |