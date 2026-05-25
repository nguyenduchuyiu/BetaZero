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
theorem algebra_bleqa_apbon2msqrtableqambsqon8b
  (a b : ℝ)
  (h₀ : 0 < a ∧ 0 < b)
  (h₁ : b ≤ a) :
  (a + b) / 2 - Real.sqrt (a * b) ≤ (a - b)^2 / (8 * b) := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h2 : 0 < a := h₀.left
    have h3 : 0 < b := h₀.right
    have h4 : b ≠ 0 := by linarith
    have h5 : a ≠ 0 := by linarith
    have h6 : (a - b) ^ 2 ≥ 0 := by
      apply sq_nonneg
    have h7 : (a + b) / (2 : ℝ) ≤ (a - b) ^ (2 : ℕ) / ((8 : ℝ) * b) + √(a * b) := by
      have h8 : (a - b) ^ (2 : ℕ) = (a - b) ^ 2 := by
        simp
      rw [h8]
      have h9 : (a - b) ^ 2 / ((8 : ℝ) * b) + √(a * b) - (a + b) / (2 : ℝ) ≥ 0 := by
        have h10 : (a - b) ^ 2 / ((8 : ℝ) * b) + √(a * b) - (a + b) / (2 : ℝ) = 
          (a ^ 2 - 6 * a * b - 3 * b ^ 2 + 8 * b * √(a * b)) / (8 * b) := by
          field_simp
          ring_nf
          <;> rw [Real.sq_sqrt (by positivity)]
          <;> ring
        rw [h10]
        have h11 : a ^ 2 - 6 * a * b - 3 * b ^ 2 + 8 * b * √(a * b) ≥ 0 := by
          have h12 : a ≥ b := by linarith
          have h13 : √(a * b) ≥ √(b * b) := Real.sqrt_le_sqrt (by nlinarith [mul_self_nonneg (a - b)])
          have h14 : √(b * b) = b := by
            rw [Real.sqrt_mul_self (by linarith)]
          nlinarith [sq_nonneg (a - b), Real.sq_sqrt (show 0 ≤ a * b by positivity), h12, h13, h14]
        have h15 : 8 * b > 0 := by
          linarith [h3]
        apply div_nonneg
        · linarith [h11]
        · nlinarith [h15]
      linarith
    exact h7


