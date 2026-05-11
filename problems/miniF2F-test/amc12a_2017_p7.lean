import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2017_p7 (f : ℕ+ → ℝ) (h₀ : f 1 = 2) (h₁ : ∀ (n:ℕ+), Even n → f n = f (n - 1) + 1)
  (h₂ : ∀ (n:ℕ+), 1 < n ∧ Odd (↑n:ℕ) → f n = f (n - 2) + 2) : f 2017 = 2018 := by
  sorry
