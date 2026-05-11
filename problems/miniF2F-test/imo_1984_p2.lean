import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem imo_1984_p2
  (a b : ℕ)
  (h₀ : 0 < a ∧ 0 < b)
  (h₁ : ¬ 7 ∣ a * b * (a + b))
  (h₂ : (7^7) ∣ ((a + b)^7 - a^7 - b^7)) :
  19 ≤ a + b := by
  sorry
