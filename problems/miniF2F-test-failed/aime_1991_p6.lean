import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem aime_1991_p6
  (r : ℝ)
  (h₀ : (∑ k ∈ Finset.Icc (19 : ℕ) 91, Int.floor (r + k / 100)) = 546) :
  Int.floor (100 * r) = 743 := by
  sorry
