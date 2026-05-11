import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_1124
  (n x : ℕ)
  (h₀ : Nat.digits 10 x = [n, 4, 7, 3])
  (h₁ : 18 ∣ x) : n = 4 := by
  sorry
