import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_44
  (L₁ L₂ I : Set (EuclideanSpace ℝ (Fin 2)))
  (hL₁ : L₁ = { x | x 0 = 9 - 2 * (x 1)})
  (hL₂ : L₂ = { x | x 1 = 3 * (x 0) + 1})
  (hI : I = L₁ ∩ L₂)
  (A : EuclideanSpace ℝ (Fin 2)) :
  A ∈ I ↔ A 0 = 1 ∧ A 1 = 4 := by
  sorry
