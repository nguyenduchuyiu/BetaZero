**Skeleton pipeline funnel (sum across problems):**

| stage | count |
|---|---:|
| requested | 3966 |
| raw_verify_success | 870 |
| raw_verify_failed | 3060 |
| patch_attempted | 3060 |
| patch_scored | 2788 |
| patch_failed | 278 |
| feedback_generated | 2788 |
| inserted_raw | 3930 |
| selected_by_beam | 457 |
| rejected_by_beam | 0 |
| children_new | 1096 |
| children_duplicate | 0 |

**Commitment outcomes:**

| stage | count |
|---|---:|
| committed | 457 |
| reserved | 383 |
| fallback_activated | 11 |
| committed_solved | 163 |
| committed_failed | 305 |
| committed_stale | 16289 |
| blocked_new_skeleton_due_to_active_commit | 0 |

**Rates:**

| rate | value |
|---|---:|
| raw_verify_success_rate | 21.94% |
| committed_rate | 11.52% |
| patch_score_rate | 91.11% |
| committed_solved_rate | 35.67% |