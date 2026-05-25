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
lemma amc12b_2002_p19_1
  (a b c : ℝ)
  (h₀ : (0 : ℝ) < a ∧ (0 : ℝ) < b ∧ (0 : ℝ) < c)
  (h₁ : a * (b + c) = (152 : ℝ))
  (h₂ : b * (c + a) = (162 : ℝ))
  (h₃ : c * (a + b) = (170 : ℝ)) :
  a * b * c = (720 : ℝ) := by
    have h4 : a * b + a * c = 152 := by linarith
    have h5 : b * c + b * a = 162 := by linarith
    have h6 : c * a + c * b = 170 := by linarith
    have h7 : a * b + b * c + c * a = 242 := by
      linarith
    have h8 : a * b * c = 720 := by
      nlinarith [mul_pos h₀.left h₀.right.left, mul_pos h₀.left h₀.right.right, mul_pos h₀.right.left h₀.right.right]
    exact h8