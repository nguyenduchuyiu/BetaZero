import Mathlib
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat
set_option pp.instanceTypes true
set_option pp.numericTypes true
set_option pp.coercions.types true
set_option pp.letVarTypes true
set_option pp.structureInstanceTypes true
set_option pp.instanceTypes true
set_option pp.mvars.withType true
set_option pp.coercions true
set_option pp.funBinderTypes true
set_option pp.piBinderTypes true
lemma mathd_algebra_113_1
  (x : ℝ) :
  (-46 : ℝ) ≤ x ^ (2 : ℕ) - (14 : ℝ) * x + (3 : ℝ) := by
  have h : x ^ 2 - 14 * x + 3 = (x - 7) ^ 2 - 46 := by
    ring
  rw [h]
  have h1 : 0 ≤ (x - 7) ^ 2 := by
    apply sq_nonneg
  linarith