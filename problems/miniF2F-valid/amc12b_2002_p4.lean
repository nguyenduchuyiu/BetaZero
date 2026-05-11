import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12b_2002_p4
  (n : ℕ)
  (h₀ : 0 < n)
  (h₁ : ((1:ℚ) / (2 : ℚ) + 1 / (3 : ℚ) + 1 / (7 : ℚ) + 1 / (↑n : ℚ)).isInt) :
  n = 42 := by
  sorry
