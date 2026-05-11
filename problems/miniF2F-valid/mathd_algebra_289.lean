import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_289 (k t m n : ℕ) (f : ℕ → ℤ) (hf : f = fun (x:ℕ) ↦ ((↑x:ℤ) ^ 2 - m * x + n)) (h₀ : Nat.Prime m ∧ Nat.Prime n) (h₁ : t < k)
  (h₂ : f k = 0) (h₃ : f t = 0) :
  m ^ n + n ^ m + k ^ t + t ^ k = 20 := by
  sorry
