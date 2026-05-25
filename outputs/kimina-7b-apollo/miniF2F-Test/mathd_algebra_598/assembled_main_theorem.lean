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
theorem mathd_algebra_598
  (a b c d : ℝ)
  (h₁ : ((4:ℝ)^a) = 5)
  (h₂ : ((5:ℝ)^b) = 6)
  (h₃ : ((6:ℝ)^c) = 7)
  (h₄ : ((7:ℝ)^d) = 8) :
  a * b * c * d = 3 / 2 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have ha : a = Real.logb 4 5 := by
      rw [←h₁]
      field_simp [Real.logb_eq_iff_rpow_eq]
    have hb : b = Real.logb 5 6 := by
      rw [←h₂]
      field_simp [Real.logb_eq_iff_rpow_eq]
    have hc : c = Real.logb 6 7 := by
      rw [←h₃]
      field_simp [Real.logb_eq_iff_rpow_eq]
    have hd : d = Real.logb 7 8 := by
      rw [←h₄]
      field_simp [Real.logb_eq_iff_rpow_eq]
    rw [ha, hb, hc, hd]
    have h : Real.logb 4 5 * Real.logb 5 6 * Real.logb 6 7 * Real.logb 7 8 = Real.logb 4 8 := by
      field_simp [Real.logb]
      <;> ring
    rw [h]
    have h2 : Real.logb 4 8 = (3 / 2 : ℝ) := by
      rw [Real.logb]
      have h3 : Real.log (8 : ℝ) = 3 * Real.log (2 : ℝ) := by
        rw [show (8 : ℝ) = 2 ^ (3 : ℝ) by norm_num]
        simp [Real.log_rpow]
      have h4 : Real.log (4 : ℝ) = 2 * Real.log (2 : ℝ) := by
        rw [show (4 : ℝ) = 2 ^ (2 : ℝ) by norm_num]
        simp [Real.log_rpow]
      rw [h3, h4]
      field_simp
      <;> ring
    linarith [h2]

