import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12b_2020_p22
  (S : Set ℝ)
  (hS: S = {x:ℝ | ∃ t, x = (2 ^ t - 3 * t) * t / 4 ^ t}) :
  IsGreatest S (1 / 12:ℝ) := by
  sorry
