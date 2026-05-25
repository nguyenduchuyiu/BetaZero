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
theorem aime_1999_p11
  (m : ℚ)
  (h₀ : 0 < m)
  (h₁ : ∑ k ∈ Finset.Icc (1 : ℕ) 35, Real.sin (5 * k * π / 180) = Real.tan (m * π / 180))
  (h₂ : (m.num:ℝ) / m.den < 90) :
  ↑m.den + m.num = 177 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
    sorry