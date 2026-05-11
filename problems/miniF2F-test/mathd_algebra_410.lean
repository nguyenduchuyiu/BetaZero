import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_410
  (f : ℝ → ℝ)
  (h₀ : f = fun x ↦ x ^ 2 - 6 * x + 13) :
  IsLeast (Set.range f) 4 := by
  sorry
