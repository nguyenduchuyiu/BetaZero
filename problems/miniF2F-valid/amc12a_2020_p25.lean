import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2020_p25
  (a : ℚ)
  (ha₀ : 0 < a)
  (S : Finset ℝ)
  (h₀ : ∀ (x : ℝ), x ∈ S ↔ ↑⌊x⌋ * (x - ↑⌊x⌋) = ↑a * x ^ 2)
  (h₁ : (∑ k ∈ S, k) = 420) :
  ↑a.den + a.num = 929 := by
  sorry
