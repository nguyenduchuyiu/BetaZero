import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_284
  (x : ℕ)
  (h₀ : 0 < x)
  (h₁ : (Nat.digits 10 x).length = 2)
  (h₁ : 2 * (Nat.digits 10 x).sum = x) :
  x = 18 := by
  sorry
