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
lemma algebra_sqineq_at2malt1_1
  (a : ℝ) :
  a * ((2 : ℝ) - a) ≤ (1 : ℝ) := by
    have h : a * ((2 : ℝ) - a) ≤ (1 : ℝ) := by
      have h1 : (a - 1) ^ 2 ≥ 0 := sq_nonneg (a - 1)
      linarith
    linarith