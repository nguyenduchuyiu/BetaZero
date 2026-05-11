import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_422
  (f : ℝ → ℝ)
  (hf : f = fun x ↦ 5 * x - 12)
  (x : ℝ)
  (hx : f ⁻¹' {x} = {f (x + 1)}) :
  x = 47 / 24 := by
  sorry
