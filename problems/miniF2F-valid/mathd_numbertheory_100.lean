import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_100 (n : ℕ) (h₁ : Nat.gcd n 40 = 10)
  (h₂ : Nat.lcm n 40 = 280) : n = 70 := by
  sorry
