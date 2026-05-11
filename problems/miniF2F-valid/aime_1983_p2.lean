import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem aime_1983_p2
  (p : ℝ)
  (f : ℝ → ℝ)
  (h₀ : f = fun x => abs (x - p) + abs (x - 15) + abs (x - p - 15))
  (h₁ : 0 < p ∧ p < 15)
  (S R : Set ℝ)
  (hS : S = Set.Icc p 15)
  (hR : R = {y | ∃ x ∈ S, y = f x}) :
  IsLeast R 15 := by
  sorry
