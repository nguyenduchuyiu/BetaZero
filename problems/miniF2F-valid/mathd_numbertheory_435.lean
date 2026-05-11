import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_435
  (S : Set ℕ)
  (hS: S = {k | ∀ n:ℕ, 0 < n → Nat.gcd (6 * n + k) (6 * n + 3) = 1 ∧Nat.gcd (6 * n + k) (6 * n + 2) = 1 ∧ Nat.gcd (6 * n + k) (6 * n + 1) = 1}) :
  IsLeast S 5 := by
  sorry
