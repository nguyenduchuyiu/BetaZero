import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_392 (n : ℕ) (h₀ : Even n)
  (h₁ : ↑n ^ 2 + (↑n + 2) ^ 2 + (↑n + 4) ^ 2 = (12296 : ℤ)) :
  ↑n * (↑n + 2) * (↑n + 4) / 8 = (32736 : ℤ) := by
  sorry
