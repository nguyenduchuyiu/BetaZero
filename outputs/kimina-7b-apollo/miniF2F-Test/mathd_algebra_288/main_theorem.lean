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
theorem mathd_algebra_288
  (x y : ℝ)
  (n : NNReal)
  (h₀ : x < 0 ∧ y < 0)
  (h₁ : abs y = 6)
  (h₂ : Real.sqrt ((x - 8)^2 + (y - 3)^2) = 15)
  (h₃ : Real.sqrt (x^2 + y^2) = Real.sqrt n) :
  n = 52 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
    sorry