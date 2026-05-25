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
lemma algebra_apbpceq2_abpbcpcaeq1_aleq1on3anbleq1ancleq4on3_1
  (a b c : ℝ)
  (h₀ : a ≤ b ∧ b ≤ c)
  (h₁ : a + b + c = (2 : ℝ))
  (h₂ : a * b + b * c + c * a = (1 : ℝ)) :
  (0 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) ∧ (1 / 3 : ℝ) ≤ b ∧ b ≤ (1 : ℝ) ∧ (1 : ℝ) ≤ c ∧ c ≤ (4 / 3 : ℝ) := by
  have h3 : 0 ≤ a := by nlinarith [sq_nonneg (a - 1 / 3), sq_nonneg (b - 1), sq_nonneg (c - 4 / 3), h₁, h₂, h₀.left, h₀.right]
  have h4 : a ≤ 1 / 3 := by nlinarith [sq_nonneg (a - 1 / 3), sq_nonneg (b - 1), sq_nonneg (c - 4 / 3), h₁, h₂, h₀.left, h₀.right]
  have h5 : 1 / 3 ≤ b := by nlinarith [sq_nonneg (a - 1 / 3), sq_nonneg (b - 1), sq_nonneg (c - 4 / 3), h₁, h₂, h₀.left, h₀.right]
  have h6 : b ≤ 1 := by nlinarith [sq_nonneg (a - 1 / 3), sq_nonneg (b - 1), sq_nonneg (c - 4 / 3), h₁, h₂, h₀.left, h₀.right]
  have h7 : 1 ≤ c := by nlinarith [sq_nonneg (a - 1 / 3), sq_nonneg (b - 1), sq_nonneg (c - 4 / 3), h₁, h₂, h₀.left, h₀.right]
  have h8 : c ≤ 4 / 3 := by nlinarith [sq_nonneg (a - 1 / 3), sq_nonneg (b - 1), sq_nonneg (c - 4 / 3), h₁, h₂, h₀.left, h₀.right]
  exact ⟨h3, h4, h5, h6, h7, h8⟩