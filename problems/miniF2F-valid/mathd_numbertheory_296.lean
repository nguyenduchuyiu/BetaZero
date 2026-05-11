import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_296
  (S : Set ℕ)
  (hS :  S = {n | 0 < n ∧ n ≠ 1 ∧ ∃ x, x ^ 3 = n ∧ ∃ t, t ^ 4 = n}) :
  IsLeast S 4096 := by
  sorry
