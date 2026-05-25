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
lemma amc12b_2021_p3_1
  (x : ℝ)
  (h₀ : (2 : ℝ) + ((1 : ℝ) + ((2 : ℝ) + ((3 : ℝ) + x)⁻¹ * (2 : ℝ))⁻¹)⁻¹ = (144 / 53 : ℝ)) :
  x = (3 / 4 : ℝ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry