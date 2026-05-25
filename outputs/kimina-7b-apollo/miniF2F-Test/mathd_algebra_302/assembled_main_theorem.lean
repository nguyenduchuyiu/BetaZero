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
theorem mathd_algebra_302 :
  (Complex.I / 2)^2 = -(1 / 4) := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h1 : Complex.I ^ 2 = -1 := by 
      simp [Complex.I_sq]
    calc 
      Complex.I ^ (2 : ℕ) * (1 / 4 : ℂ) = Complex.I ^ 2 * (1 / 4 : ℂ) := by 
        simp
      _ = (-1) * (1 / 4 : ℂ) := by 
        rw [h1]
      _ = (-1 / 4 : ℂ) := by 
        ring

