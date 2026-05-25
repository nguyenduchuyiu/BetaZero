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
theorem aime_1983_p2
  (x p : ℝ)
  (f : ℝ → ℝ)
  (h₀ : 0 < p ∧ p < 15)
  (h₁ : p ≤ x ∧ x ≤ 15)
  (h₂ : f x = abs (x - p) + abs (x - 15) + abs (x - p - 15)) :
  15 ≤ f x := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have hp1 : 0 < p := h₀.left
    have hp2 : p < 15 := h₀.right
    have hpx1 : p ≤ x := h₁.left
    have hpx2 : x ≤ 15 := h₁.right
    have h1 : x - p ≥ 0 := by linarith
    have h2 : x - 15 ≤ 0 := by linarith
    have h3 : x - p - 15 ≤ 0 := by linarith
    have h_abs1 : |x - p| = x - p := abs_of_nonneg h1
    have h_abs2 : |(-15 : ℝ) + x| = 15 - x := by
      rw [abs_of_nonpos (by linarith)]
      ring
    have h_abs3 : |(-15 : ℝ) + (x - p)| = - (x - p) + 15 := by
      have h : (-15 : ℝ) + (x - p) ≤ 0 := by linarith
      rw [abs_of_nonpos h]
      ring
    have h : |x - p| + |(-15 : ℝ) + x| + |(-15 : ℝ) + (x - p)| = -x + 30 := by
      rw [h_abs1, h_abs2, h_abs3]
      linarith
    linarith [h, hpx2]

