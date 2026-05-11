import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem algebra_ineq_nto1onlt2m1on
  (n : ℕ)
  (hn : 0 < n) :
  (n : ℝ) ^ (1 / n:ℝ) < 2 - 1 / n := by
  sorry
