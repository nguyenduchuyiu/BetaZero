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
lemma mathd_algebra_484_1:
  Real.log (27 : ℝ) / Real.log (3 : ℝ) = (3 : ℝ) := by
  have h1 : Real.log (27 : ℝ) = 3 * Real.log (3 : ℝ) := by
    have h2 : (27 : ℝ) = (3 : ℝ) ^ (3 : ℝ) := by norm_num
    rw [h2]
    simp [Real.log_rpow]
  rw [h1]
  field_simp