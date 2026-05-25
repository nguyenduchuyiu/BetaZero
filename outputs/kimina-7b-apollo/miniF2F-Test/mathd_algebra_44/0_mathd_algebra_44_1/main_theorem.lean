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
lemma mathd_algebra_44_1
  (s t : ℝ)
  (h₀ : s = (9 : ℝ) - (2 : ℝ) * t)
  (h₁ : t = (3 : ℝ) * s + (1 : ℝ)) :
  s = (1 : ℝ) ∧ t = (4 : ℝ) := by
  constructor
  · -- Prove s = 1
    linarith [h₀, h₁]
  · -- Prove t = 4
    linarith [h₀, h₁]