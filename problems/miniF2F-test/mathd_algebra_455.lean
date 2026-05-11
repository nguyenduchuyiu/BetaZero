import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_455
  (f : ℕ+ → ℕ)
  (h₀: ∀ x:ℕ+, f (x + 1) = 2 * f (x))
  (h₁: f 5 = 48) :
  f 1 = 3 := by
  sorry
