# Rollout Analysis Report

## Design Intent

The rollout JSON encodes an asymmetric AND/OR proof graph. OR nodes are Lean proof states; AND nodes are generated actions. Tactics are terminal actions, while skeletons create subgoal states. `r_env` is the immediate syntactic/repair survival reward, `r_dep` is dependency-aware skeleton quality, and `Q/V` are deterministic AND/OR backups.

## Dataset Health

- Theorems: 50
- Solved roots: 12 (24.0%)
- Nodes: 22641 total, 5079 OR, 17562 AND
- Edges: 23682
- Avg nodes/theorem: 452.82
- Avg max depth: 1.74
- Shared OR states: 679
- Cycles detected: 42

## Reward Behavior

- r_env: mean=0.3153, median=0.0000, zeros=8973, ones=2257
- r_dep: mean=0.0018, median=0.0000, zeros=17529, ones=28
- Q_value: mean=0.6915, median=0.8571, zeros=3664, ones=7169
- FAILED actions with full `r_env`: 454
- Q backup mismatches: 0

## Graph Behavior

- Largest graphs:
  - `algebra_apbpceq2_abpbcpcaeq1_aleq1on3anbleq1ancleq4on3.json`: nodes=919, max_depth=3, status=OPEN
  - `amc12_2001_p21.json`: nodes=807, max_depth=3, status=OPEN
  - `aime_1995_p7.json`: nodes=795, max_depth=3, status=OPEN
  - `algebra_ineq_nto1onlt2m1on.json`: nodes=791, max_depth=3, status=OPEN
  - `aime_1984_p7.json`: nodes=765, max_depth=3, status=OPEN

## Diagnostic & Efficiency Analysis

- `missing_extracted_code` (747): An action has no parsed Lean code. The raw model output may be malformed, outside the expected code block format, or parser-incompatible.
- `failed_full_r_env` (454): Valid but incomplete tactic: `r_env` reached 1.0 on a `FAILED` action. The code is syntactically/logically sound (survival), but didn't close the goal. This represents a search inefficiency where the agent 'stalls' at high-reward partial progress.
- `cycle_detected` (42): Graph contains a directed cycle. This usually means decomposition generated a repeated state/goal path; extraction may avoid infinite recursion, but the search has spent budget revisiting itself.
