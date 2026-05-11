import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_530
  (S : Set ℕ)
  (hS : ∀ x:ℕ, x ∈ S ↔ ∃ n k :ℕ, 0 < n ∧ 0 < k ∧ (5 : ℝ) < n / k ∧ (n : ℝ) / k < 6 ∧ x = Nat.lcm n k / Nat.gcd n k) :
  IsLeast S 22 := by
  sorry
