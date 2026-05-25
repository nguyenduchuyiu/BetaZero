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
lemma algebra_amgm_sumasqdivbgeqsuma_1
  (a b c d : ℝ)
  (h₀ : (0 : ℝ) < a ∧ (0 : ℝ) < b ∧ (0 : ℝ) < c ∧ (0 : ℝ) < d) :
  a + b + c + d ≤ a ^ (2 : ℕ) / b + b ^ (2 : ℕ) / c + c ^ (2 : ℕ) / d + d ^ (2 : ℕ) / a := by
  rcases h₀ with ⟨ha, hb, hc, hd⟩
  have h1 : a ^ 2 / b + b ≥ 2 * a := by
    have h1 : a ^ 2 / b + b - 2 * a ≥ 0 := by
      have h2 : a ^ 2 / b + b - 2 * a = (a - b) ^ 2 / b := by
        field_simp
        ring
      rw [h2]
      have h3 : (a - b) ^ 2 ≥ 0 := by
        exact sq_nonneg (a - b)
      have h4 : (a - b) ^ 2 / b ≥ 0 := by
        apply div_nonneg
        · exact h3
        · linarith [hb]
      linarith
    linarith
  have h2 : b ^ 2 / c + c ≥ 2 * b := by
    have h2 : b ^ 2 / c + c - 2 * b ≥ 0 := by
      have h3 : b ^ 2 / c + c - 2 * b = (b - c) ^ 2 / c := by
        field_simp
        ring
      rw [h3]
      have h4 : (b - c) ^ 2 ≥ 0 := by
        exact sq_nonneg (b - c)
      have h5 : (b - c) ^ 2 / c ≥ 0 := by
        apply div_nonneg
        · exact h4
        · linarith [hc]
      linarith
    linarith
  have h3 : c ^ 2 / d + d ≥ 2 * c := by
    have h3 : c ^ 2 / d + d - 2 * c ≥ 0 := by
      have h4 : c ^ 2 / d + d - 2 * c = (c - d) ^ 2 / d := by
        field_simp
        ring
      rw [h4]
      have h5 : (c - d) ^ 2 ≥ 0 := by
        exact sq_nonneg (c - d)
      have h6 : (c - d) ^ 2 / d ≥ 0 := by
        apply div_nonneg
        · exact h5
        · linarith [hd]
      linarith
    linarith
  have h4 : d ^ 2 / a + a ≥ 2 * d := by
    have h4 : d ^ 2 / a + a - 2 * d ≥ 0 := by
      have h5 : d ^ 2 / a + a - 2 * d = (d - a) ^ 2 / a := by
        field_simp
        ring
      rw [h5]
      have h6 : (d - a) ^ 2 ≥ 0 := by
        exact sq_nonneg (d - a)
      have h7 : (d - a) ^ 2 / a ≥ 0 := by
        apply div_nonneg
        · exact h6
        · linarith [ha]
      linarith
    linarith
  linarith