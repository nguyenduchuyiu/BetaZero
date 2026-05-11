import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12_2000_p12
  (S: Set ℕ)
  (hS: S = {x | ∃ a m c : ℕ, (x = a * m * c + a * m + m * c + a * c) ∧a + m + c = 12}) :
  IsGreatest S 112 := by
  sorry
