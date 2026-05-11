import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_293
  (n x : ℕ)
  (h₀ : Nat.digits 10 x = [7, n, 0, 2])
  (h₁ : 11 ∣ x) :
  n = 5 := by
  sorry
