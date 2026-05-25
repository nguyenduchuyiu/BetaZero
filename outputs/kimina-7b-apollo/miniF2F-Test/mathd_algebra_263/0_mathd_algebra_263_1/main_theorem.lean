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
lemma mathd_algebra_263_1
  (y : ℝ)
  (h₀ : (0 : ℝ) ≤ (19 : ℝ) + (3 : ℝ) * y)
  (h₁ : √((19 : ℝ) + (3 : ℝ) * y) = (7 : ℝ)) :
  y = (10 : ℝ) := by
    have h2 : (19 : ℝ) + (3 : ℝ) * y = (49 : ℝ) := by
      calc
        (19 : ℝ) + (3 : ℝ) * y = (√((19 : ℝ) + (3 : ℝ) * y)) ^ 2 := by
          rw [Real.sq_sqrt]
          linarith
        _ = (7 : ℝ) ^ 2 := by rw [h₁]
        _ = (49 : ℝ) := by norm_num
    linarith