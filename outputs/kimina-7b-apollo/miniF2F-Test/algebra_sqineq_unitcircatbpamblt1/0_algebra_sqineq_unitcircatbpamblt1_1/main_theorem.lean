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
lemma algebra_sqineq_unitcircatbpamblt1_1
  (a b : ℝ)
  (h₀ : a ^ (2 : ℕ) + b ^ (2 : ℕ) = (1 : ℝ)) :
  a * b + (a - b) ≤ (1 : ℝ) := by
    have h1 : a ≤ 1 := by
      nlinarith [sq_nonneg (a - 1), sq_nonneg b, h₀]
    have h2 : b ≤ 1 := by
      nlinarith [sq_nonneg (b - 1), sq_nonneg a, h₀]
    have h3 : -1 ≤ a := by
      nlinarith [sq_nonneg (a + 1), sq_nonneg b, h₀]
    have h4 : -1 ≤ b := by
      nlinarith [sq_nonneg (b + 1), sq_nonneg a, h₀]
    nlinarith [sq_nonneg (a - 1), sq_nonneg (b + 1), h₀]