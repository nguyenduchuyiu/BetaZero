import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_277
  (S: Set ℕ)
  (hS: S = {x | ∃ m n:ℕ, x = m + n ∧ Nat.gcd m n = 6 ∧ Nat.lcm m n = 126}):
  IsLeast S 60 := by
  sorry
