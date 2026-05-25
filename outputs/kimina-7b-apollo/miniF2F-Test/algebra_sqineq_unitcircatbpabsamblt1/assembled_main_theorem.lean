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
theorem algebra_sqineq_unitcircatbpabsamblt1
  (a b: ℝ)
  (h₀ : a^2 + b^2 = 1) :
  a * b + |a - b| ≤ 1 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    by_cases h : a - b ≥ 0
    · -- Case 1: a - b ≥ 0, which means a ≥ b
      rw [abs_of_nonneg h]
      have h1 : a ^ 2 + b ^ 2 = 1 := by
        norm_num at h₀
        linarith
      nlinarith [sq_nonneg (a - 1), sq_nonneg (b), sq_nonneg (a - b), h1, sq_nonneg (a + b - 1), sq_nonneg (a - 1 / 2), sq_nonneg (b - 1 / 2)]
    · -- Case 2: a - b < 0, which means a < b
      rw [abs_of_neg (lt_of_not_ge h)]
      have h1 : a ^ 2 + b ^ 2 = 1 := by
        norm_num at h₀
        linarith
      nlinarith [sq_nonneg (a - 1), sq_nonneg (b), sq_nonneg (a - b), h1, sq_nonneg (a + b - 1), sq_nonneg (a - 1 / 2), sq_nonneg (b - 1 / 2)]

