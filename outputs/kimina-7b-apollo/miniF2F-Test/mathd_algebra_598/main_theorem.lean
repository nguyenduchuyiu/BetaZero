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
theorem mathd_algebra_598
  (a b c d : ℝ)
  (h₁ : ((4:ℝ)^a) = 5)
  (h₂ : ((5:ℝ)^b) = 6)
  (h₃ : ((6:ℝ)^c) = 7)
  (h₄ : ((7:ℝ)^d) = 8) :
  a * b * c * d = 3 / 2 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
    sorry