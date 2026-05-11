import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2011_p18
  (S : Set ℝ)
  (hS : S = {z | ∃ x y:ℝ, abs (x + y) + abs (x - y) = 2 ∧ z = x ^ 2 - 6 * x + y ^ 2}) :
  IsGreatest S 8 := by
  sorry
