import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_113
  (f : ℝ → ℝ)
  (hf: f = fun x => x^2 - 14 * x + 3) :
  IsMinOn f (Set.univ) 7 := by
  sorry
