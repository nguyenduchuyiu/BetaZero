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
lemma imo_1964_p2_1
  (a b c : ℝ)
  (h₀ : (0 : ℝ) < a ∧ (0 : ℝ) < b ∧ (0 : ℝ) < c)
  (h₁ : c < a + b)
  (h₂ : b < a + c)
  (h₃ : a < b + c) :
  a ^ (2 : ℕ) * (b + c - a) + b ^ (2 : ℕ) * (c + a - b) + c ^ (2 : ℕ) * (a + b - c) ≤ (3 : ℝ) * a * b * c := by
  nlinarith [sq_nonneg (a - b), sq_nonneg (b - c), sq_nonneg (c - a),
    sq_nonneg (a - b), sq_nonneg (b - c), sq_nonneg (c - a),
    mul_pos h₀.left h₀.right.left, mul_pos h₀.left h₀.right.right, mul_pos h₀.right.left h₀.right.right]