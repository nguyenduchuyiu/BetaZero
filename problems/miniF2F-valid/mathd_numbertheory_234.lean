import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_234
  (a b n : ℕ)
  (h₀ : Nat.digits 10 n = [b, a])
  (h₁ : n ^ 3 = 912673) : 
  a + b = 16 := by
  sorry
