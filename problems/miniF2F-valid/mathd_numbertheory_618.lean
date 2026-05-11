import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_618
  (p : ℕ → ℕ)
  (S : Set ℕ)
  (h₀ : ∀ x, p x = x ^ 2 - x + 41)
  (h₁ : S = {n | 1 < Nat.gcd (p n) (p (n + 1))}) :
  IsLeast S 41 := by
  sorry
