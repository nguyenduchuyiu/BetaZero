import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem imo_1983_p6
  (T : Affine.Triangle ℝ (EuclideanSpace ℝ (Fin 2))) :
  let a := dist (T.points 1) (T.points 2)
  let b := dist (T.points 0) (T.points 2)
  let c := dist (T.points 0) (T.points 1)
  0 ≤ a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ∧
  (0 = a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ↔
  (a = b ∧ a = c)) := by
  sorry
