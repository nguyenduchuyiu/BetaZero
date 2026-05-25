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
lemma mathd_algebra_33_1
  (x y z : ℝ)
  (h₀ : ¬x = (0 : ℝ))
  (h₂ : y * (7 : ℝ) = z * (10 : ℝ))
  (h₁ : x * (2 : ℝ) = y * (5 : ℝ)) :
  z * x⁻¹ = (7 / 25 : ℝ) := by
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