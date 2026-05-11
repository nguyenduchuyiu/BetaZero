import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem imo_1962_p4
  (S : Set ℝ)
  (h₀ : S = {x : ℝ | (Real.cos x)^2 + (Real.cos (2 * x))^2 + (Real.cos (3 * x))^2 = 1}) :
  S = {x : ℝ | ∃ k : ℤ, (x = π / 2 + k * π) ∨ (x = π / 4 + k * π / 2)
  ∨ (x = π / 6 + k * π) ∨ (x = 5 * π / 6 + k * π)} := by
  sorry
