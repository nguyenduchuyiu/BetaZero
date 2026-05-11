import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_150
  (S : Set ℕ)
  (hS: S = {n:ℕ | 0 < n ∧ ¬ Nat.Prime (7 + 30 * n)}) :
  IsLeast S 6 := by
  sorry
