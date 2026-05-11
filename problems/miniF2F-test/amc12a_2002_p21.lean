import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2002_p21
  (u S : ℕ → ℕ)
  (h₀ : u 0 = 4)
  (h₁ : u 1 = 7)
  (h₂ : ∀ n ≥ 2, u n = (u (n - 2) + u (n - 1)) % 10)
  (h₃ : ∀ n, S n = ∑ k ∈ Finset.range n, u k)
  (Sn : Set ℕ)
  (h₄ : Sn = {n | S n > 10000}) :
  IsLeast Sn 1999 := by
  sorry
