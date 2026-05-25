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
lemma mathd_algebra_156_1
  (x y : ℝ)
  (f g : ℝ → ℝ)
  (h₀ : ∀ (t : ℝ), f t = t ^ (4 : ℕ))
  (h₄ : x ^ (2 : ℕ) < y ^ (2 : ℕ))
  (h₃ : y ^ (4 : ℕ) = (-6 : ℝ) + y ^ (2 : ℕ) * (5 : ℝ))
  (h₂ : x ^ (4 : ℕ) = (-6 : ℝ) + x ^ (2 : ℕ) * (5 : ℝ))
  (h₁ : ∀ (t : ℝ), g t = (-6 : ℝ) + t ^ (2 : ℕ) * (5 : ℝ)) :
  y ^ (2 : ℕ) - x ^ (2 : ℕ) = (1 : ℝ) := by
    have hy2 : (y ^ (2 : ℕ)) ^ 2 - 5 * (y ^ (2 : ℕ)) + 6 = 0 := by
      have h₃' : y ^ (4 : ℕ) = (-6 : ℝ) + y ^ (2 : ℕ) * (5 : ℝ) := h₃
      simp [pow_succ, pow_zero, mul_comm] at h₃'
      linarith
    have hx2 : (x ^ (2 : ℕ)) ^ 2 - 5 * (x ^ (2 : ℕ)) + 6 = 0 := by
      have h₂' : x ^ (4 : ℕ) = (-6 : ℝ) + x ^ (2 : ℕ) * (5 : ℝ) := h₂
      simp [pow_succ, pow_zero, mul_comm] at h₂'
      linarith
    have hy2_eq : (y ^ (2 : ℕ) - 2) * (y ^ (2 : ℕ) - 3) = 0 := by
      have h : (y ^ (2 : ℕ)) ^ 2 - 5 * (y ^ (2 : ℕ)) + 6 = 0 := hy2
      linarith
    have hx2_eq : (x ^ (2 : ℕ) - 2) * (x ^ (2 : ℕ) - 3) = 0 := by
      have h : (x ^ (2 : ℕ)) ^ 2 - 5 * (x ^ (2 : ℕ)) + 6 = 0 := hx2
      linarith
    have hy2_val : y ^ (2 : ℕ) = 3 := by
      cases' (mul_eq_zero.mp hy2_eq) with hy2_sub2 hy2_sub3
      · -- Case y^2 - 2 = 0, so y^2 = 2
        have : y ^ (2 : ℕ) = 2 := by linarith
        cases' (mul_eq_zero.mp hx2_eq) with hx2_sub2 hx2_sub3
        · -- Case x^2 - 2 = 0, so x^2 = 2
          have : x ^ (2 : ℕ) = 2 := by linarith
          linarith [sq_nonneg (y ^ (2 : ℕ) - x ^ (2 : ℕ)), h₄]
        · -- Case x^2 - 3 = 0, so x^2 = 3
          have : x ^ (2 : ℕ) = 3 := by linarith
          linarith [sq_nonneg (y ^ (2 : ℕ) - x ^ (2 : ℕ)), h₄]
      · -- Case y^2 - 3 = 0, so y^2 = 3
        linarith
    have hx2_val : x ^ (2 : ℕ) = 2 := by
      cases' (mul_eq_zero.mp hx2_eq) with hx2_sub2 hx2_sub3
      · -- Case x^2 - 2 = 0, so x^2 = 2
        linarith
      · -- Case x^2 - 3 = 0, so x^2 = 3
        have : x ^ (2 : ℕ) = 3 := by linarith
        linarith [sq_nonneg (y ^ (2 : ℕ) - x ^ (2 : ℕ)), h₄]
    have h_diff : y ^ (2 : ℕ) - x ^ (2 : ℕ) = (1 : ℝ) := by
      rw [hy2_val, hx2_val]
      norm_num
    exact h_diff