import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_5
  (S : Set ℕ)
  (hS : S = {x | 10 < x ∧ (∃ y:ℕ, y ^ 2 = x) ∧ (∃ z:ℕ, z ^ 3 = x)}):
  IsLeast S 64 := by
  sorry
