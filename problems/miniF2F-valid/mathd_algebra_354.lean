import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_354
  (f : ℕ+ → ℝ)
  (h₀ : ∃ d, ∀ n, f (n + 1) = f n + d)
  (h₁ : f (7) = 30)
  (h₂ : f 11 = 60) :
  f 21 = 135 := by
  sorry
