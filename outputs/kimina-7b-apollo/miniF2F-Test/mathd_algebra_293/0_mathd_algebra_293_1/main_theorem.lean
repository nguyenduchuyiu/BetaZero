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
lemma mathd_algebra_293_1
  (x : NNReal) :
  √(60 : ℝ) * √(↑x : ℝ) * (√(12 : ℝ) * √(↑x : ℝ)) * (√(63 : ℝ) * √(↑x : ℝ)) =
    (36 : ℝ) * (↑x : ℝ) * (√(35 : ℝ) * √(↑x : ℝ)) := by
  have h1 : √(60 : ℝ) * √(↑x : ℝ) * (√(12 : ℝ) * √(↑x : ℝ)) * (√(63 : ℝ) * √(↑x : ℝ)) =
    √(60 : ℝ) * √(12 : ℝ) * √(63 : ℝ) * (√(↑x : ℝ) * √(↑x : ℝ) * √(↑x : ℝ)) := by
    ring_nf
  rw [h1]
  have h2 : √(60 : ℝ) * √(12 : ℝ) * √(63 : ℝ) = √(60 * 12 * 63 : ℝ) := by
    rw [← Real.sqrt_mul (by norm_num)]
    rw [← Real.sqrt_mul (by norm_num)]
    all_goals norm_num
  have h3 : √(↑x : ℝ) * √(↑x : ℝ) * √(↑x : ℝ) = (√(↑x : ℝ) ^ 2) * √(↑x : ℝ) := by ring
  rw [h2, h3]
  have h4 : √(↑x : ℝ) ^ 2 = (↑x : ℝ) := by
    rw [Real.sq_sqrt]
    positivity
  rw [h4]
  have h5 : √(60 * 12 * 63 : ℝ) * (↑x : ℝ) * √(↑x : ℝ) = (36 : ℝ) * (↑x : ℝ) * (√(35 : ℝ) * √(↑x : ℝ)) := by
    have h6 : √(60 * 12 * 63 : ℝ) = 36 * √(35 : ℝ) := by
      rw [Real.sqrt_eq_iff_sq_eq] <;> norm_num
      <;> ring_nf
      <;> norm_num
      <;> ring_nf
      <;> norm_num
    rw [h6]
    ring
  linarith [h5]