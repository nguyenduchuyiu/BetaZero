import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_765
  (S : Set ℤ)
  (hS : S = {x | x < 0 ∧ (24 * x) ≡ 15 [ZMOD 1199]}) :
  IsGreatest S (-449:ℤ) := by
  sorry
