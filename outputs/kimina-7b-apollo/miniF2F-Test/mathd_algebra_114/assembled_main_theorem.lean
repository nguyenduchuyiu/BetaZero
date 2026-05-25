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
theorem mathd_algebra_114
  (a : ℝ)
  (h₀ : a = 8) :
  (16 * (a^2) ^ (1 / 3 : ℝ)) ^ (1 / 3 : ℝ) = 4 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h1 : (64 : ℝ) ^ (1 / 3 : ℝ) = (4 : ℝ) := by
      rw [show (64 : ℝ) = (4 ^ (3 : ℝ)) by norm_num]
      rw [← Real.rpow_mul]
      norm_num
      all_goals norm_num
    rw [h1]
    have h2 : (4 * (16 : ℝ)) ^ (1 / 3 : ℝ) = (4 : ℝ) := by
      have h3 : (4 * (16 : ℝ)) = (64 : ℝ) := by norm_num
      rw [h3]
      have h4 : (64 : ℝ) ^ (1 / 3 : ℝ) = (4 : ℝ) := by
        rw [show (64 : ℝ) = (4 ^ (3 : ℝ)) by norm_num]
        rw [← Real.rpow_mul]
        norm_num
        all_goals norm_num
      exact h4
    exact h2

