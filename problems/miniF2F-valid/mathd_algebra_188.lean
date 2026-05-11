import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_188
  (f : ℝ → ℝ)
  (h₀ : Function.Bijective f)
  (h₁ : f 2 = 4)
  (h₂ : f 2 = Function.invFun f 2) :
  f (f 2) = 2 := by
  sorry
