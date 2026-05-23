import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem imo_1977_p5
  (a b q r : ℕ)
  (hp: 0 < a ∧ 0 < b)
  (h₀ : r = (a ^ 2 + b ^ 2) % (a + b))
  (h₁ : q = (a ^ 2 + b ^ 2) / (a + b))
  (h₂ : q ^ 2 + r = 1977) :
  (a = 7 ∧ b = 50) ∨ (a = 37 ∧ b = 50) ∨ (a = 50 ∧ b = 7) ∨ (a = 50 ∧ b = 37) := by
  sorry
