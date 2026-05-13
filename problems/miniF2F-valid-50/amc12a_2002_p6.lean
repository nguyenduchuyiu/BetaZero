import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2002_p6
  (S: Set ℕ)
  (hS : S = {m | 0 < m ∧ (∃ n, 0 < n ∧ m * n ≤ m + n)}) :
  Set.Infinite S := by
  sorry
