import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_559
  (S: Set ℕ)
  (hS: S = {x | x % 3 = 2 ∧ ∃ y:ℕ, (y % 5 = 4 ∧ x % 10 = y % 10)}) :
  IsLeast S 14 := by
  sorry
