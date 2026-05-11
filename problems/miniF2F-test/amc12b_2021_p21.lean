import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12b_2021_p21
  (S : Finset ℝ)
  (a : ℝ)
  (h₀ : ∀ x : ℝ, x ∈ S ↔ 0 < x ∧ x ^ (2 : ℝ) ^ Real.sqrt 2 = Real.sqrt 2 ^ ((2 : ℝ) ^ x))
  (h₁ : a = ∑ k ∈ S, k) :
  (2 ≤ a) ∧ a < 6 := by
  sorry
