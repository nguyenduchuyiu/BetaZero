import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem imo_1973_p3
  (S : Set (ℝ × ℝ))
  (hS : S = {(a, b) | ∃ x : ℝ, x ^ 4 + a * x ^ 3 + b * x ^ 2 + a * x + 1 = 0}) :
  IsLeast {z | ∃ a b : ℝ, (a, b) ∈ S ∧ z = a ^ 2 + b ^ 2} (4 / 5) := by
  sorry
