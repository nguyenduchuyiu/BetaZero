import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_342
  (a d : ℝ)
  (h₀ : ∑ k ∈ Finset.range 5, (a + ↑k * d) = 70)
  (h₁ : ∑ k ∈ Finset.range 10, (a + (↑k:ℝ) * d) = (210)) :
  a = 42 / 5 := by
  sorry
