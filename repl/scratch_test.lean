import Mathlib

open BigOperators Nat Real Topology

theorem my_theorem (a b : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : a * b = a - b) : a / b + b / a - a * b = 2 := by
  have ha : a ≠ 0 := h₀.left
  have hb : b ≠ 0 := h₀.right
  have h_ab_nonzero : a * b ≠ 0 := by
    exact mul_ne_zero ha hb
  have h_frac_sum : a / b + b / a = (a ^ 2 + b ^ 2) / (a * b) := by
    field_simp [ha, hb]
  have h_target_rewrite : a / b + b / a - a * b = ((a ^ 2 + b ^ 2) / (a * b)) - a * b := by
    rw [h_frac_sum]
  have h_main : ((a ^ 2 + b ^ 2) / (a * b)) - a * b = 2 := by
    field_simp [h_ab_nonzero]
    have h2 : (a - b)^2 = (a * b)^2 := by rw [h₁]
    ring_nf at h2
    ring_nf
    linarith [h2, h₁]
  have h_final : a / b + b / a - a * b = 2 := by
    rw [h_target_rewrite, h_main]
  exact h_final
