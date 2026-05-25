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
theorem amc12b_2002_p19
  (a b c: ℝ)
  (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : a * (b + c) = 152)
  (h₂ : b * (c + a) = 162)
  (h₃ : c * (a + b) = 170) :
  a * b * c = 720 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h4 : a * b + a * c = 152 := by linarith
    have h5 : b * c + b * a = 162 := by linarith
    have h6 : c * a + c * b = 170 := by linarith
    have h7 : a * b + b * c + c * a = 242 := by
      linarith
    have h8 : a * b * c = 720 := by
      nlinarith [mul_pos h₀.left h₀.right.left, mul_pos h₀.left h₀.right.right, mul_pos h₀.right.left h₀.right.right]
    exact h8

