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
theorem mathd_algebra_44
  (s t : ℝ)
  (h₀ : s = 9 - 2 * t)
  (h₁ : t = 3 * s + 1) :
  s = 1 ∧ t = 4 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    constructor
    · -- Prove s = 1
      linarith [h₀, h₁]
    · -- Prove t = 4
      linarith [h₀, h₁]

