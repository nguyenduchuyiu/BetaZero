import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2021_p25
  (m : ℕ)
  (hm₀ : 0 < m)
  (f : ℕ → ℝ)
  (h₀ : ∀ n, 0 < n → f n = ((Nat.divisors n).card)/(n^((1:ℝ)/3)))
  (h₁ : ∀ n ≠ m, 0 < n → f n < f m) :
  (Nat.digits 10 m).sum = 9 := by
  sorry
