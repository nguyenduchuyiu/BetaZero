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
lemma mathd_algebra_137_1
  (x : ℕ)
  (h₀ : (↑x : ℝ) + (4 / 100 : ℝ) * (↑x : ℝ) = (598 : ℝ)) :
  x = (575 : ℕ) := by
    have h1 : (x : ℝ) = (598 : ℝ) * (25 / 26) := by
      linarith
    have h2 : (x : ℝ) = (575 : ℝ) := by
      rw [h1]
      norm_num
    exact_mod_cast h2