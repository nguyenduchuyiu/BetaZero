import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_215
  (S : Finset ℝ)
  (h₀ : ∀ x : ℝ, x ∈ S ↔ (x + 3) ^ 2 = 121) :
  (∑ k ∈ S, k) = -6 ∧ S.card = 2 := by
  sorry
