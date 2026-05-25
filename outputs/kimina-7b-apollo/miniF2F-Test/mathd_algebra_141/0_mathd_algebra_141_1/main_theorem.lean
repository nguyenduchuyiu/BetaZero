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
lemma mathd_algebra_141_1
  (a b : ℝ)
  (h₁ : a * b = (180 : ℝ))
  (h₂ : (2 : ℝ) * (a + b) = (54 : ℝ)) :
  a ^ (2 : ℕ) + b ^ (2 : ℕ) = (369 : ℝ) := by
  have h3 : a + b = (27 : ℝ) := by
    linarith
  have h4 : a ^ 2 + b ^ 2 = (369 : ℝ) := by
    have h5 : a ^ 2 + b ^ 2 = (a + b) ^ 2 - 2 * (a * b) := by
      ring
    rw [h5, h3, h₁]
    norm_num
  exact h4