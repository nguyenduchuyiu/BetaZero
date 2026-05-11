import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_487
  (F G I : Set (EuclideanSpace ℝ (Fin 2)))
  (hF : F = { x | x 1 = (x 0) ^ 2})
  (hG : G = { x | x 0 + x 1 = 1})
  (hI : I = (F ∩ G))
  (A B : EuclideanSpace ℝ (Fin 2))
  (h₀ : ∀ x, x ∈ I ↔ x = A ∨ x = B) :
  dist A B = Real.sqrt 10 := by sorry
  sorry
