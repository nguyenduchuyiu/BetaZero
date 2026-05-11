import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_323 
  (f : ℝ → ℝ)
  (h₀: f = fun x ↦ x ^ 3 - 8) :
  Function.invFun f (f (Function.invFun f 19)) = 3 := by
  sorry
