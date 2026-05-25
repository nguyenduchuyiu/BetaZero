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
lemma mathd_algebra_129_1
  (a : ℝ)
  (h₀ : ¬a = (0 : ℝ))
  (h₁ : (1 / 2 : ℝ) - a⁻¹ = (1 : ℝ)) :
  a = (-2 : ℝ) := by
    have h2 : a⁻¹ = (1 / 2 : ℝ) - (1 : ℝ) := by linarith
    have h3 : a⁻¹ = (-1 / 2 : ℝ) := by
      rw [h2]
      norm_num
    have h4 : a = (-2 : ℝ) := by
      have h5 : a ≠ (0 : ℝ) := h₀
      field_simp at h3
      linarith
    exact h4