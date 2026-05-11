import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_28
  (S : Set ℝ)
  (hS : S = {c | ∃ x:ℝ, 2 * x ^ 2 + 5 * x + c = 0}) :
  IsGreatest S (25/8:ℝ) := by
  sorry
