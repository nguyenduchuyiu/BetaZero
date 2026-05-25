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
lemma mathd_algebra_338_1
  (a b c : ℝ)
  (h₀ : (3 : ℝ) * a + b + c = (-3 : ℝ))
  (h₁ : a + (3 : ℝ) * b + c = (9 : ℝ))
  (h₂ : a + b + (3 : ℝ) * c = (19 : ℝ)) :
  a * b * c = (-56 : ℝ) := by
  have h3 : a + b + c = 5 := by
    linarith
  have ha : a = -4 := by
    linarith [h₀, h3]
  have hb : b = 2 := by
    linarith [h₁, h3]
  have hc : c = 7 := by
    linarith [h₂, h3]
  rw [ha, hb, hc]
  norm_num