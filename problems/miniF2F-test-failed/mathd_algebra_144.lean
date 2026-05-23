import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_144
  (S : Finset ℕ)
  (hS : ∀ n:ℕ, n ∈ S ↔ ∃ a b c : ℕ , 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 60 ∧ ∃d:ℕ, 0 < d ∧ (a = b + d ∧ b = c + d)) :
  S.card = 9 := by
  sorry
