import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_270 (f : ℝ → ℝ) (h₀ : ∀ (x : ℝ), f x = 1 / (x + 2)) :
  f (f 1) = 3 / 7 := by
  sorry
