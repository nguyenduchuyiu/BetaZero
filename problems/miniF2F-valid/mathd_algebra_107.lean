import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_107
  (x y : ℝ)
  (h₀ : x ^ 2 + 8 * x + y ^ 2 - 6 * y = 0) :
  ∃ a b r:ℝ, 0 < r ∧ (x + a) ^ 2 + (y + b) ^ 2 = r ^ 2 ∧ r = 5 := by
  sorry
