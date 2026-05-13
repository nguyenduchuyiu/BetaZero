import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem aime_1994_p3 (f : ℤ → ℤ) (h₁ : ∀ x, f x + f (x-1) = x^2) (h₂ : f 19 = 94) : f 94 % 1000 = 561 := by
  sorry
