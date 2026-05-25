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
lemma mathd_algebra_270_1
  (f : ℝ → ℝ)
  (h₀ : ∀ (x : ℝ), ¬x = (-2 : ℝ) → f x = ((2 : ℝ) + x)⁻¹) :
  f (f (1 : ℝ)) = (3 / 7 : ℝ) := by
    have h1 : f (1 : ℝ) = (1 / 3 : ℝ) := by
      have h1 : f (1 : ℝ) = ((2 : ℝ) + (1 : ℝ))⁻¹ := h₀ 1 (by norm_num)
      rw [h1]
      norm_num
    have h2 : f ((1 / 3 : ℝ)) = (3 / 7 : ℝ) := by
      have h2 : f ((1 / 3 : ℝ)) = ((2 : ℝ) + (1 / 3 : ℝ))⁻¹ := h₀ (1 / 3) (by norm_num)
      rw [h2]
      norm_num
    rw [h1]
    exact h2