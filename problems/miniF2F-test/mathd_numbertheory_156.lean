import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_156
  (S : Set ℕ)
  (hS : S = {x | ∃ n, 0 < n ∧ x = Nat.gcd (n + 7) (2 * n + 1)}) :
  IsGreatest S 13 := by
  sorry
