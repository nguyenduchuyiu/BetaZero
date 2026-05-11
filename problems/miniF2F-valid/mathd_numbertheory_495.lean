import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_495
  (S : Set ℕ)
  (hS : S = {n | ∃ a b:ℕ, 0 < a ∧ 0 < b ∧ a % 10 = 2 ∧ b % 10 = 4 ∧ Nat.gcd a b = 6 ∧ n = Nat.lcm a b}) :
  IsLeast S 108 := by
  sorry
