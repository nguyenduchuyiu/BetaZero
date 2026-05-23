import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2020_p13 (a b c : ℕ) (h₁ : 1 < a ∧ 1 < b ∧ 1 < c)
  (h₂ : ∀ n:NNReal, n ≠ 1 → (n * (n * n ^ (1 / c)) ^ (1 / b)) ^ (1 / a) = (n ^ 25) ^ (1 / 36)) : b = 3 := by
  sorry
