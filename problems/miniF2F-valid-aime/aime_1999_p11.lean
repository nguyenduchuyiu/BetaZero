import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem aime_1999_p11
  (m n : ℕ)
  (hm₀ : 0 < m)
  (hn₀ : 0 < n)
  (h₀ : Nat.Coprime m n)
  (h₁ : (∑ k ∈ Finset.Icc (1 : ℕ) 35, Real.sin (5 * k * Real.pi / 180)) = Real.tan ((m / n:ℝ) * Real.pi / 180))
  (h₂ : (m : ℝ) / (n : ℝ) < 90) :
  m + n = 177 := by
  sorry
