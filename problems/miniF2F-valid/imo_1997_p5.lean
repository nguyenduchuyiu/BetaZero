import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem imo_1997_p5 (x y : ℕ) (h₀ : 1 ≤ x ∧ 1 ≤ y) (h₁ : x ^ y ^ 2 = y ^ x) :
  (x, y) = (1, 1) ∨ (x, y) = (16, 2) ∨ (x, y) = (27, 3) := by
  sorry
