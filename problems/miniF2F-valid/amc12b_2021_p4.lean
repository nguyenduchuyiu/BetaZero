import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12b_2021_p4
  (m a : ℕ)
  (gm ga : ℕ → ℝ)
  (h₀ : 0 < m ∧ 0 < a)
  (h₁ : (↑m:ℝ) / ↑a = (3 : ℝ) / 4)
  (h₂: ∑ x ∈ Finset.range m, gm x / (m:ℝ) = 84)
  (h₂: ∑ x ∈ Finset.range a, ga x / (a:ℝ) = 70) :
  (∑ x ∈ Finset.range m, gm x + ∑ x ∈ Finset.range a, ga x) / ((↑m:ℝ) + ↑a) = (76 : ℝ) := by
  sorry
