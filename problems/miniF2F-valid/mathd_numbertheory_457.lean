import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_457
  (S : Set ℕ)
  (hS: S = {n | 0 < n ∧  80325 ∣ Nat.factorial n}) :
  IsLeast S 17 := by
  sorry
