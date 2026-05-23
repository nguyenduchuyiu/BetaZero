import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12b_2002_p6 
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : a ≠ 0 ∧ b ≠ 0)
  (h₁: f = fun x => x ^ 2 + a * x + b)
  (h₂: f a = 0)
  (h₃ : f b = 0) : 
  a = 1 ∧ b = -2 := by
  sorry
