import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2021_p18
  (f : ℚ → ℝ)
  (h₀ : ∀ (x y : ℚ), 0 < x → 0 < y → f (x * y) = f x + f y)
  (h₁ : ∀ (p : ℕ), (Nat.Prime p) → f p = p) :
  f (25 / 11) < 0 := by
  sorry
