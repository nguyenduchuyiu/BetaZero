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
theorem mathd_algebra_33
  (x y z : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : 2 * x = 5 * y)
  (h₂ : 7 * y = 10 * z) :
  z / x = 7 / 25 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have hy : y ≠ 0 := by
      by_contra h
      rw [h] at h₁
      have : x = 0 := by linarith
      contradiction
    have hz : z = (7 / 10 : ℝ) * y := by
      linarith
    have hyx : y = (2 / 5 : ℝ) * x := by
      field_simp [h₀] at *
      linarith
    have hzx : z = (7 / 25 : ℝ) * x := by
      rw [hz, hyx]
      ring
    field_simp [h₀]
    rw [hzx]
    ring

