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
lemma amc12a_2009_p6_1
  (m n p q : ℝ)
  (h₀ : p = (2 : ℝ) ^ m)
  (h₁ : q = (3 : ℝ) ^ n) :
  ((2 : ℝ) ^ m) ^ (n * (2 : ℝ)) * ((3 : ℝ) ^ n) ^ m = (12 : ℝ) ^ (n * m) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry