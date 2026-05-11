import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_184
  (a b : NNReal)
  (h₀ : 0 < a ∧ 0 < b)
  (S₁ S₂ : Fin 3 → NNReal)
  (h₁ : S₁ 0 = 6 ∧ S₁ 1 = a ∧ S₁ 2 = b)
  (h₂ : S₂ 0 = 1 / b ∧ S₁ 1 = a ∧ S₁ 2 = 54)
  (h₃ : ∃! r, r ≠ 0 ∧ r ≠ 1 ∧ ∀ n, n < 2 → S₁ n * r = S₁ (n + 1))
  (h₄ : ∃! r, r ≠ 0 ∧ r ≠ 1 ∧ ∀ n, S₂ n * r = S₂ (n + 1)) :
  a = 3 * NNReal.sqrt 2 := by
  sorry
