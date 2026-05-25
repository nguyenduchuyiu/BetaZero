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
lemma mathd_algebra_80_1
  (x : ℝ)
  (h₀ : ¬x = (-1 : ℝ))
  (h₁ : x * ((1 : ℝ) + x)⁻¹ - ((1 : ℝ) + x)⁻¹ * (9 : ℝ) = (2 : ℝ)) :
  x = (-11 : ℝ) := by
  have h2 : (1 : ℝ) + x ≠ 0 := by
    intro h
    have : x = -1 := by linarith
    contradiction
  have eq1 : x - 9 = 2 * (1 + x) := by
    have h3 : ((1 : ℝ) + x) ≠ 0 := h2
    field_simp [h3] at h₁
    linarith
  linarith