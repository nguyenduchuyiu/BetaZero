import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_403 
  (h₀: (∑ k ∈ Nat.properDivisors 18, k) = 21) : 
  (∑ k ∈ Nat.properDivisors 198, k) = 270 := by
  sorry
