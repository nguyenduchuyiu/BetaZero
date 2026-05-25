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
theorem mathd_algebra_270
  (f : ℝ → ℝ)
  (h₀ : ∀ x, x ≠ -2 -> f x = 1 / (x + 2)) :
  f (f 1) = 3/7 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
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

