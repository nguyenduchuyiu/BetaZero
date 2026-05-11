import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_329
  (L₁ L₂ I : Set (EuclideanSpace ℝ (Fin 2)))
  (hL₁ : L₁ = { x | x 0 = 3 * (x 1)})
  (hL₂ : L₂ = { x | 2 * (x 0) + 5 * (x 1) = 11})
  (hI : I = L₁ ∩ L₂)
  (A : EuclideanSpace ℝ (Fin 2))
  (hA : A ∈ I) :
  A 0 + A 1 = 4 := by
  sorry
