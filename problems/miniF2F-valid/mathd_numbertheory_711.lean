import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_711
  (S: Set ℕ)
  (hS : S = {x | ∃ m n, x = m + n ∧ Nat.gcd m n = 8 ∧ Nat.lcm m n = 112}):
  IsLeast S 72 := by
  sorry
