import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem imo_2006_p3
  (a b c : ℝ)
  (S : Set ℝ)
  (hS : S = {M | a * b * (a ^ 2 - b ^ 2) + b * c * (b ^ 2 - c ^ 2) + c * a * (c ^ 2 - a ^ 2) ≤ M * (a ^ 2 + b ^ 2 + c ^ 2) ^ 2}) :
  IsLeast S (9 * Real.sqrt 2 / 32) := by
  sorry
