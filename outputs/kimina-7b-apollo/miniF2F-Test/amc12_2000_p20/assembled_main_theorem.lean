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
theorem amc12_2000_p20
  (x y z : ℝ)
  (h₀ : 0 < x ∧ 0 < y ∧ 0 < z)
  (h₁ : x + 1/y = 4)
  (h₂ : y + 1/z = 1)
  (h₃ : z + 1/x = 7/3) :
  x*y*z = 1 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h4 : x * y * z + (x * y * z)⁻¹ = 2 := by
      have h5 : x ≠ 0 := by linarith [h₀.left]
      have h6 : y ≠ 0 := by linarith [h₀.right.left]
      have h7 : z ≠ 0 := by linarith [h₀.right.right]
      have eq1 : (x + y⁻¹) * (y + z⁻¹) * (z + x⁻¹) = (4 : ℝ) * (1 : ℝ) * (7 / 3 : ℝ) := by
        rw [h₁, h₂, h₃]
      have eq2 : (x + y⁻¹) * (y + z⁻¹) * (z + x⁻¹) = x * y * z + (x * y * z)⁻¹ + (x + y⁻¹ + y + z⁻¹ + z + x⁻¹) := by
        field_simp [h5, h6, h7]
        ring
      rw [eq2] at eq1
      have eq3 : x + y⁻¹ + y + z⁻¹ + z + x⁻¹ = (4 : ℝ) + (1 : ℝ) + (7 / 3 : ℝ) := by
        linarith [h₁, h₂, h₃]
      rw [eq3] at eq1
      nlinarith [eq1]
    have h7 : x * y * z > 0 := by
      apply mul_pos
      apply mul_pos
      all_goals linarith [h₀.left, h₀.right.left, h₀.right.right]
    have h8 : x * y * z = 1 := by
      have h9 : (x * y * z) * ((x * y * z)⁻¹) = 1 := by
        field_simp [h7.ne.symm]
      nlinarith [h4, h9, h7]
    exact h8

