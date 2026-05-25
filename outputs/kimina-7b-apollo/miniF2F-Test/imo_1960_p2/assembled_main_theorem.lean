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
theorem imo_1960_p2
  (x : ℝ)
  (h₀ : 0 ≤ 1 + 2 * x)
  (h₁ : (1 - Real.sqrt (1 + 2 * x))^2 ≠ 0)
  (h₂ : (4 * x^2) / (1 - Real.sqrt (1 + 2*x))^2 < 2*x + 9)
  (h₃ : x ≠ 0) :
  -(1 / 2) ≤ x ∧ x < 45 / 8 := by


  
  have h4 : x ≥ -(1 / 2 : ℝ) := by
    have h5 : (0 : ℝ) ≤ (1 : ℝ) + (2 : ℝ) * x := h₀
    linarith
  have h6 : x < (45 / 8 : ℝ) := by
    have h7 : ((1 : ℝ) - √((1 : ℝ) + (2 : ℝ) * x)) ^ (2 : ℕ) ≠ 0 := h₁
    have h8 : (4 : ℝ) * x ^ (2 : ℕ) / ((1 : ℝ) - √((1 : ℝ) + (2 : ℝ) * x)) ^ (2 : ℕ) < (2 : ℝ) * x + (9 : ℝ) := h₂
    have h9 : (4 : ℝ) * x ^ (2 : ℕ) < ((2 : ℝ) * x + (9 : ℝ)) * ((1 : ℝ) - √((1 : ℝ) + (2 : ℝ) * x)) ^ (2 : ℕ) := by
      have h10 : ((1 : ℝ) - √((1 : ℝ) + (2 : ℝ) * x)) ^ (2 : ℕ) > 0 := by
        by_contra h
        push_neg at h
        have : ((1 : ℝ) - √((1 : ℝ) + (2 : ℝ) * x)) ^ (2 : ℕ) = 0 := by linarith [sq_nonneg ((1 : ℝ) - √((1 : ℝ) + (2 : ℝ) * x))]
        contradiction
      apply (div_lt_iff (by positivity)).mp at h8
      linarith
    have h11 : ((1 : ℝ) - √((1 : ℝ) + (2 : ℝ) * x)) ^ (2 : ℕ) = (1 : ℝ) - 2 * √((1 : ℝ) + (2 : ℝ) * x) + (√((1 : ℝ) + (2 : ℝ) * x)) ^ (2 : ℕ) := by
      ring
    have h12 : (√((1 : ℝ) + (2 : ℝ) * x)) ^ (2 : ℕ) = (1 : ℝ) + (2 : ℝ) * x := by
      rw [Real.sq_sqrt]
      linarith
    rw [h11, h12] at h9
    nlinarith [sq_nonneg (x - (45 / 8 : ℝ)), sq_nonneg (√((1 : ℝ) + (2 : ℝ) * x) - (7 / 2 : ℝ)), h9, Real.sqrt_nonneg ((1 : ℝ) + (2 : ℝ) * x), h4, h8]
  exact ⟨h4, h6⟩

