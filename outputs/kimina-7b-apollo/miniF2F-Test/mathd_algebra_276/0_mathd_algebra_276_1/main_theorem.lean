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
lemma mathd_algebra_276_1
  (a b : ℤ)
  (h₀ : ∀ (x : ℝ), (10 : ℝ) * x ^ (2 : ℕ) - x - (24 : ℝ) = ((↑a : ℝ) * x - (8 : ℝ)) * ((↑b : ℝ) * x + (3 : ℝ))) :
  a * b + b = (12 : ℤ) := by
    have h1 := h₀ 1
    have h2 := h₀ (-1)
    have h3 := h₀ 2
    norm_num at h1 h2 h3
    have h4 : (a : ℝ) * (b : ℝ) = 10 := by
      have h_eq1 := h₀ 2
      have h_eq2 := h₀ (-2)
      norm_num at h_eq1 h_eq2
      nlinarith [sq_nonneg ((a : ℝ) * 2 - (b : ℝ) * 8), sq_nonneg ((a : ℝ) - (b : ℝ) * 3)]
    have h5 : (3 : ℝ) * (a : ℝ) - 8 * (b : ℝ) = -1 := by
      have h_eq1 := h₀ 1
      have h_eq2 := h₀ (-1)
      norm_num at h_eq1 h_eq2
      nlinarith [sq_nonneg ((a : ℝ) - (b : ℝ) * 3), sq_nonneg ((a : ℝ) + (b : ℝ) * 8)]
    have h6 : a * b = 10 := by
      norm_cast at h4
    have h7 : 3 * a - 8 * b = -1 := by
      norm_cast at h5
    have : a = 5 ∧ b = 2 := by
      have h8 : a * b = 10 := h6
      have h9 : 3 * a - 8 * b = -1 := h7
      have h10 : a ≤ 10 := by nlinarith [h8]
      have h11 : b ≤ 10 := by nlinarith [h8]
      have h12 : a ≥ -10 := by nlinarith [h8]
      have h13 : b ≥ -10 := by nlinarith [h8]
      interval_cases a <;> omega
    rcases this with ⟨ha, hb⟩
    rw [ha, hb]
    norm_num