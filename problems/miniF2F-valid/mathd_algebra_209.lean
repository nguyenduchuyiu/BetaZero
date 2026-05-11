import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_209
  (f h : ℝ → ℝ)
  (h₀: Function.LeftInverse f h)
  (h₁ : h 2 = 10)
  (h₂ : h 10 = 1)
  (h₃ : h 1 = 2) :
  f (f 10) = 1 := by
  sorry
