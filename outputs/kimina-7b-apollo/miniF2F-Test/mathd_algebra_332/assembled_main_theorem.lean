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
theorem mathd_algebra_332
  (x y : ℝ)
  (h₀ : (x + y) / 2 = 7)
  (h₁ : Real.sqrt (x * y) = Real.sqrt 19) :
  x^2 + y^2 = 158 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h2 : x + y = 14 := by
      linarith
    have h3 : x * y = 19 := by
      have h3 : x * y ≥ 0 := by
        have h4 : √(x * y) = √(19 : ℝ) := h₁
        have h5 : x * y ≥ 0 := by
          by_contra h
          push_neg at h
          have h6 : √(x * y) = 0 := by
            rw [Real.sqrt_eq_zero']
            all_goals linarith
          have h7 : √(19 : ℝ) = 0 := by
            linarith [h6, h4]
          have h8 : (19 : ℝ) = 0 := by
            rw [Real.sqrt_eq_zero'] at h7
            all_goals linarith
          norm_num at h8
        linarith
      have h4 : x * y = 19 := by
        have h5 : √(x * y) = √(19 : ℝ) := h₁
        have h6 : x * y = 19 := by
          have h7 : √(x * y) ^ 2 = √(19 : ℝ) ^ 2 := by rw [h5]
          rw [Real.sq_sqrt (by linarith)] at h7
          rw [Real.sq_sqrt (by norm_num)] at h7
          linarith
        linarith
      linarith
    have h4 : x ^ 2 + y ^ 2 = (158 : ℝ) := by
      have h5 : x ^ 2 + y ^ 2 = (x + y) ^ 2 - 2 * (x * y) := by
        ring
      rw [h5, h2, h3]
      norm_num
    exact h4

