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
theorem algebra_sqineq_unitcircatbpamblt1
  (a b: ℝ)
  (h₀ : a^2 + b^2 = 1) :
  a * b + (a - b) ≤ 1 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h1 : a ≤ 1 := by
      nlinarith [sq_nonneg (a - 1), sq_nonneg b, h₀]
    have h2 : b ≤ 1 := by
      nlinarith [sq_nonneg (b - 1), sq_nonneg a, h₀]
    have h3 : -1 ≤ a := by
      nlinarith [sq_nonneg (a + 1), sq_nonneg b, h₀]
    have h4 : -1 ≤ b := by
      nlinarith [sq_nonneg (b + 1), sq_nonneg a, h₀]
    nlinarith [sq_nonneg (a - 1), sq_nonneg (b + 1), h₀]

