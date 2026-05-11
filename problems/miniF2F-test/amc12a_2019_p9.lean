import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2019_p9
  (a : ℕ → ℚ) (h₀ : a 1 = 1) (h₁ : a 2 = 3 / 7)
  (h₂ : ∀ n, 3 ≤ n → a n = (a (n - 2) * a (n - 1)) / (2 * a (n - 2) - a (n - 1))) :
  ↑(a 2019).den + (a 2019).num = 8078 := by
  sorry
