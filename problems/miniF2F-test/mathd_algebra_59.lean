import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_59
  (b : ℝ)
  (F : ℝ → ℝ → ℝ → ℝ → ℝ)
  (h₀: F = fun a b c d ↦ a ^ b + c ^ d)
  (h₁ : F 4 b 2 3 = 12) :
  b = 1 := by
  sorry
