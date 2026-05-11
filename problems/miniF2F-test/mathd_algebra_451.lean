import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_451
  (f g : ℝ → ℝ)
  (hg : Function.RightInverse g f)
  (h₀ : g (-15) = 0) (h₁ : g 0 = 3) (h₂ : g 3 = 9)
  (h₃ : g 9 = 20) : f (f 9) = 0 := by
  sorry
