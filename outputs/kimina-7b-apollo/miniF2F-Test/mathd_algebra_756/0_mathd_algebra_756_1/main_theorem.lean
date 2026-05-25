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
lemma mathd_algebra_756_1
  (a b : ℝ)
  (h₀ : (2 : ℝ) ^ a = (32 : ℝ))
  (h₁ : a ^ b = (125 : ℝ)) :
  b ^ a = (243 : ℝ) := by
  have ha : a = (Real.logb (2 : ℝ) (32 : ℝ)) := by
    rw [← h₀]
    field_simp
  have ha' : a = (5 : ℝ) := by
    rw [ha]
    have h2 : (32 : ℝ) = (2 ^ (5 : ℝ) : ℝ) := by norm_num
    rw [h2]
    rw [Real.logb_rpow (by norm_num) (by norm_num)]
  rw [ha'] at h₁
  have hb : b = (Real.logb (5 : ℝ) (125 : ℝ)) := by
    rw [← h₁]
    field_simp
  have hb' : b = (3 : ℝ) := by
    rw [hb]
    have h2 : (125 : ℝ) = (5 ^ (3 : ℝ) : ℝ) := by norm_num
    rw [h2]
    rw [Real.logb_rpow (by norm_num) (by norm_num)]
  rw [hb', ha']
  norm_num