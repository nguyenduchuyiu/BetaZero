import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_22
  (S : Finset ℕ)
  (hS : S = {b | ∃ x y : ℕ , Nat.digits 10 x = [6, b] ∧ 0 < y ∧ y ^ 2 = x}) :
  S.card = 2 := by
  sorry
