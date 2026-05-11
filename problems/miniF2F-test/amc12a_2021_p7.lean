import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2021_p7
  (S : Set ℝ)
  (hS : S = {z | ∃ x y:ℝ, z = (x * y - 1) ^ 2 + (x + y) ^ 2}) :
  IsLeast S 1 := by
  sorry
