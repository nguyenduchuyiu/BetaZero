import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_234
  (a : ℕ → ℚ)
  (d : ℚ)
  (h₀: ∀ n, a (n + 1) = a n * d)
  (h₁: a 0 = 27 / 125)
  (h₂: a 1 = 9 / 25)
  (h₃: a 2 = 3 / 5) :
  a 5 = 25 / 9 := by
  sorry
