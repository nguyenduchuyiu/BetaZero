import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_214 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h₀ : ∀ x, f x = a * (x) ^ 2 + b * x + c)
  (h₁: IsExtrOn f Set.univ 2)
  (h₂: f 2 = 3)
  (h₃ : f 4 = 4) :
  f 6 = 7 := by
  sorry
