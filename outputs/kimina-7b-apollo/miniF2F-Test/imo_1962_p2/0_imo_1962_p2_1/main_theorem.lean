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
lemma imo_1962_p2_1
  (x : ℝ)
  (h₀ : (0 : ℝ) ≤ (3 : ℝ) - x)
  (h₁ : (0 : ℝ) ≤ x + (1 : ℝ))
  (h₂ : (1 / 2 : ℝ) < √((3 : ℝ) - x) - √(x + (1 : ℝ))) :
  (-1 : ℝ) ≤ x ∧ x < (1 : ℝ) - √(31 : ℝ) / (8 : ℝ) := by
    have h3 : -1 ≤ x := by linarith
    have h4 : x < (1 : ℝ) - √(31 : ℝ) / (8 : ℝ) := by
      have h3 : √((3 : ℝ) - x) > √(x + (1 : ℝ)) + 1 / 2 := by linarith
      have h4 : √(x + (1 : ℝ)) ≥ 0 := Real.sqrt_nonneg (x + (1 : ℝ))
      have h5 : √((3 : ℝ) - x) ^ 2 > (√(x + (1 : ℝ)) + 1 / 2) ^ 2 := by nlinarith [h3, h4]
      have h6 : √((3 : ℝ) - x) ^ 2 = (3 : ℝ) - x := by
        rw [Real.sq_sqrt]
        linarith
      have h7 : (√(x + (1 : ℝ)) + 1 / 2) ^ 2 = (√(x + (1 : ℝ))) ^ 2 + 2 * √(x + (1 : ℝ)) * (1 / 2) + (1 / 2) ^ 2 := by ring
      have h8 : (√(x + (1 : ℝ))) ^ 2 = x + (1 : ℝ) := by
        rw [Real.sq_sqrt]
        linarith
      rw [h6, h7, h8] at h5
      have h9 : 3 - x > (x + 1) + √(x + (1 : ℝ)) + (1 / 2) ^ 2 := by linarith
      have h10 : (1 / 2 : ℝ) ^ 2 = (1 / 4 : ℝ) := by norm_num
      rw [h10] at h9
      have h11 : √(x + (1 : ℝ)) ≥ 0 := Real.sqrt_nonneg (x + (1 : ℝ))
      have h12 : ((7 / 4 : ℝ) - 2 * x) ^ 2 = (49 / 16 : ℝ) - (28 / 4 : ℝ) * x + (4 : ℝ) * x ^ 2 := by ring
      have h13 : √(x + (1 : ℝ)) < (7 / 4) - 2 * x := by
        nlinarith [h9, Real.sqrt_nonneg (x + (1 : ℝ))]
      have h14 : √(x + (1 : ℝ)) ^ 2 < ((7 / 4) - 2 * x) ^ 2 := by nlinarith [h13, Real.sqrt_nonneg (x + (1 : ℝ))]
      have h15 : (7 / 4 : ℝ) - 2 * x > 0 := by nlinarith [h13, Real.sqrt_nonneg (x + (1 : ℝ))]
      have h16 : √(x + (1 : ℝ)) ^ 2 = x + (1 : ℝ) := by
        rw [Real.sq_sqrt]
        linarith
      have h17 : x + (1 : ℝ) < ((7 / 4 : ℝ) - 2 * x) ^ 2 := by nlinarith [h14, h16]
      have h18 : ((7 / 4 : ℝ) - 2 * x) ^ 2 = (49 / 16 : ℝ) - (28 / 4 : ℝ) * x + (4 : ℝ) * x ^ 2 := by ring
      rw [h18] at h17
      have h19 : 4 * x ^ 2 - 8 * x + (33 / 16 : ℝ) > 0 := by nlinarith [h17]
      have h20 : x < (8 - √31) / 8 := by
        have h21 : 4 * x ^ 2 - 8 * x + (33 / 16 : ℝ) > 0 := h19
        have h22 : (x - (8 - √31) / 8) * (4 * x - (8 + √31) / 2) > 0 := by
          ring_nf
          nlinarith [Real.sq_sqrt (show 0 ≤ (31 : ℝ) by norm_num), h21]
        cases mul_pos_iff.mp h22 with
        | inl h23 =>
          have h24 : x > (8 - √31) / 8 := by linarith
          have h25 : x > (8 + √31) / 8 := by
            have h26 : (8 + √31) / 8 < x := by linarith
            linarith
          have h27 : (8 + √31) / 8 ≥ 1 := by
            have h28 : √31 ≥ 5 := by
              have h29 : (5 : ℝ) ^ 2 < (31 : ℝ) := by norm_num
              have h30 : 0 ≤ √31 := Real.sqrt_nonneg 31
              have h31 : 5 ≤ √31 := by
                apply Real.le_sqrt_of_sq_le
                linarith
              linarith
            linarith
          linarith
        | inr h23 =>
          have h24 : x < (8 - √31) / 8 := by linarith
          linarith
      linarith
    exact ⟨h3, h4⟩