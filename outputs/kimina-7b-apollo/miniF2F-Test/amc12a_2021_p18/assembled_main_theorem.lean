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
theorem amc12a_2021_p18
  (f : ℚ → ℝ)
  (h₀ : ∀x>0, ∀y>0, f (x * y) = f x + f y)
  (h₁ : ∀p, Nat.Prime p → f p = p) :
  f (25 / 11) < 0 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h2 : f (25 / 11 : ℚ) = f (5^2 * (11 : ℚ)⁻¹) := by
      norm_num
    rw [h2]
    have h3 : f (5^2 * (11 : ℚ)⁻¹) = f (5^2) + f ((11 : ℚ)⁻¹) := by
      specialize h₀ (5^2 : ℚ) (by norm_num) ((11 : ℚ)⁻¹) (by norm_num)
      exact h₀
    rw [h3]
    have h4 : f (5^2 : ℚ) = f (5 : ℚ) + f (5 : ℚ) := by
      specialize h₀ (5 : ℚ) (by norm_num) (5 : ℚ) (by norm_num)
      norm_num at h₀ ⊢
      linarith
    have h5 : f (5 : ℚ) = (5 : ℝ) := by
      specialize h₁ 5 (by norm_num)
      norm_num at h₁ ⊢
      exact_mod_cast h₁
    have h6 : f (5^2 : ℚ) = (10 : ℝ) := by
      rw [h4, h5]
      norm_num
    have h7 : f ((11 : ℚ)⁻¹) = - (11 : ℝ) := by
      have h8 : f (11 : ℚ) = (11 : ℝ) := by
        specialize h₁ 11 (by norm_num)
        norm_num at h₁ ⊢
        exact_mod_cast h₁
      have h9 : f (11 : ℚ) + f ((11 : ℚ)⁻¹) = f (1 : ℚ) := by
        specialize h₀ (11 : ℚ) (by norm_num) ((11 : ℚ)⁻¹) (by norm_num)
        norm_num at h₀ ⊢
        linarith
      have h10 : f (1 : ℚ) = (0 : ℝ) := by
        have h11 : f (1 : ℚ) = f (1 : ℚ) + f (1 : ℚ) := by
          specialize h₀ (1 : ℚ) (by norm_num) (1 : ℚ) (by norm_num)
          norm_num at h₀ ⊢
          linarith
        linarith
      rw [h10] at h9
      linarith [h8]
    rw [h6, h7]
    norm_num

