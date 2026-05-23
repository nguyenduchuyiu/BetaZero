import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2003_p25
  (f : ℝ → ℝ → ℝ → ℝ)
  (S : Finset ℝ)
  (h₀ : f = fun a b x ↦ Real.sqrt (a * x ^ 2 + b * x))
  (hS : S = {a | ∃ b, 0 < b ∧ Set.range (f a b) = { x | 0 ≤ f x }}):
  S.card = 2 := by
  sorry
