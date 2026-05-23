import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem imo_1978_p5
  (n : ℕ)
  (f : ℕ → ℕ)
  (h₀ : ∀ (m : ℕ), 0 < m → 0 < f m)
  (h₁ : ∀ (p q : ℕ), 0 < p → 0 < q → p ≠ q → f p ≠ f q)
  (h₂ : 0 < n) :
  (∑ k ∈ Finset.Icc 1 n, (1 : ℝ) / k) ≤ ∑ k ∈ Finset.Icc 1 n, ((f k):ℝ) / k ^ 2 := by
  sorry
