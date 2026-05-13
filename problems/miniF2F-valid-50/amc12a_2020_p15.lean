import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2020_p15
  (A B : Set ℂ)
  (D : Set ℝ)
  (hA : A = {x:ℂ | x ^ 3 - 8 = 0})
  (hB : B = {x:ℂ | x ^ 3 - 8 * x ^ 2 - 8 * x + 64 = 0})
  (hD : D = {d:ℝ | ∃ a b:ℂ, a ∈ A ∧ b ∈ B ∧ d = Complex.normSq (a - b)}):
  IsGreatest D ((2:ℝ) * Real.sqrt 21) := by
  sorry
