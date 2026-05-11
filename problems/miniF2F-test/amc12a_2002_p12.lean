import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2002_p12 :
  ∃! k, ∃ a b:ℕ, (Nat.Prime a ∧ Nat.Prime b ∧ a ≠ b ∧ (∀ f : ℝ → ℝ, (f = fun x:ℝ ↦ x ^ 2 - 63 * x + k ) → f (a:ℝ) = 0 ∧ f (b:ℝ) = 0)) := by
  sorry
