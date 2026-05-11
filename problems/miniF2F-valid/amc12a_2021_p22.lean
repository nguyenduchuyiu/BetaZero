import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2021_p22 (a b c : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = x ^ 3 + a * x ^ 2 + b * x + c)
  (h₁ : f (Real.cos (2 * Real.pi / 7)) = 0 ∧ f (Real.cos (4 * Real.pi / 7)) = 0 ∧ f (Real.cos (6 * Real.pi / 7)) = 0) :
  a * b * c = 1 / 32 := by
  sorry
