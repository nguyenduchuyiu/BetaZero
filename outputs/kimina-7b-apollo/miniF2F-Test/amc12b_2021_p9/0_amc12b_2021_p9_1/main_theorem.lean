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
lemma amc12b_2021_p9_1:
  Real.log (80 : ℝ) / Real.log (2 : ℝ) / (Real.log (2 : ℝ) / Real.log (40 : ℝ)) -
      Real.log (160 : ℝ) / Real.log (2 : ℝ) / (Real.log (2 : ℝ) / Real.log (20 : ℝ)) =
    (2 : ℝ) := by
  have h1 : Real.log (80 : ℝ) = Real.log (2^4 * 5) := by norm_num
  have h2 : Real.log (40 : ℝ) = Real.log (2^3 * 5) := by norm_num
  have h3 : Real.log (160 : ℝ) = Real.log (2^5 * 5) := by norm_num
  have h4 : Real.log (20 : ℝ) = Real.log (2^2 * 5) := by norm_num
  rw [h1, h2, h3, h4]
  have h5 : Real.log (2^4 * 5) = 4 * Real.log (2 : ℝ) + Real.log (5 : ℝ) := by
    rw [Real.log_mul (by norm_num) (by norm_num)]
    simp [Real.log_pow]
  have h6 : Real.log (2^3 * 5) = 3 * Real.log (2 : ℝ) + Real.log (5 : ℝ) := by
    rw [Real.log_mul (by norm_num) (by norm_num)]
    simp [Real.log_pow]
  have h7 : Real.log (2^5 * 5) = 5 * Real.log (2 : ℝ) + Real.log (5 : ℝ) := by
    rw [Real.log_mul (by norm_num) (by norm_num)]
    simp [Real.log_pow]
  have h8 : Real.log (2^2 * 5) = 2 * Real.log (2 : ℝ) + Real.log (5 : ℝ) := by
    rw [Real.log_mul (by norm_num) (by norm_num)]
    simp [Real.log_pow]
  rw [h5, h6, h7, h8]
  have h9 : Real.log (2 : ℝ) ≠ 0 := by
    have h10 : Real.log (2 : ℝ) > 0 := by
      apply Real.log_pos
      norm_num
    linarith
  field_simp [h9]
  ring