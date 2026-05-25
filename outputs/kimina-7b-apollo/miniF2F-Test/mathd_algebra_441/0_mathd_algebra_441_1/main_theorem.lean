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
lemma mathd_algebra_441_1
  (x : ℝ)
  (h₀ : ¬x = (0 : ℝ)) :
  x ^ (4 : ℕ) * x⁻¹ ^ (4 : ℕ) * (10 : ℝ) = (10 : ℝ) := by
    have h1 : x ^ (4 : ℕ) * x⁻¹ ^ (4 : ℕ) = 1 := by
      simp [pow_succ, pow_zero, h₀]
      <;> field_simp [h₀]
      <;> ring
    rw [h1]
    all_goals norm_num