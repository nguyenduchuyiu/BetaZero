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
theorem amc12b_2020_p13 :
  Real.sqrt (Real.log 6 / Real.log 2 + Real.log 6 / Real.log 3) = Real.sqrt (Real.log 3 / Real.log 2) + Real.sqrt (Real.log 2 / Real.log 3) := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h1 : Real.log (2 : ℝ) > 0 := by
      apply Real.log_pos
      norm_num
    have h2 : Real.log (3 : ℝ) > 0 := by
      apply Real.log_pos
      norm_num
    have h3 : Real.log (6 : ℝ) > 0 := by
      apply Real.log_pos
      norm_num
    have h4 : Real.log (2 : ℝ) ≠ 0 := by
      linarith [h1]
    have h5 : Real.log (3 : ℝ) ≠ 0 := by
      linarith [h2]
    have h6 : Real.log (6 : ℝ) ≠ 0 := by
      linarith [h3]
    have h7 : Real.log (6 : ℝ) = Real.log (2 : ℝ) + Real.log (3 : ℝ) := by
      rw [show (6 : ℝ) = 2 * 3 by norm_num]
      exact Real.log_mul (by linarith) (by linarith)
    have h8 : √(Real.log (6 : ℝ) / Real.log (2 : ℝ) + Real.log (6 : ℝ) / Real.log (3 : ℝ)) = √(Real.log (3 : ℝ) / Real.log (2 : ℝ)) + √(Real.log (2 : ℝ) / Real.log (3 : ℝ)) := by
      have h9 : Real.log (6 : ℝ) / Real.log (2 : ℝ) + Real.log (6 : ℝ) / Real.log (3 : ℝ) = (Real.log (3 : ℝ) / Real.log (2 : ℝ)) + (Real.log (2 : ℝ) / Real.log (3 : ℝ)) + 2 := by
        have h10 : Real.log (6 : ℝ) = Real.log (2 : ℝ) + Real.log (3 : ℝ) := h7
        have h11 : Real.log (2 : ℝ) ≠ 0 := h4
        have h12 : Real.log (3 : ℝ) ≠ 0 := h5
        field_simp [h11, h12]
        nlinarith [h10, Real.log_pos (show (2 : ℝ) > 1 by norm_num), Real.log_pos (show (3 : ℝ) > 1 by norm_num)]
      have h13 : √(Real.log (6 : ℝ) / Real.log (2 : ℝ) + Real.log (6 : ℝ) / Real.log (3 : ℝ)) = √((Real.log (3 : ℝ) / Real.log (2 : ℝ)) + (Real.log (2 : ℝ) / Real.log (3 : ℝ)) + 2) := by
        rw [h9]
      have h14 : Real.log (3 : ℝ) / Real.log (2 : ℝ) ≥ 0 := by
        apply div_nonneg
        · apply Real.log_nonneg
          norm_num
        · linarith [h1]
      have h15 : Real.log (2 : ℝ) / Real.log (3 : ℝ) ≥ 0 := by
        apply div_nonneg
        · apply Real.log_nonneg
          norm_num
        · linarith [h2]
      have h16 : √((Real.log (3 : ℝ) / Real.log (2 : ℝ)) + (Real.log (2 : ℝ) / Real.log (3 : ℝ)) + 2) = √(Real.log (3 : ℝ) / Real.log (2 : ℝ)) + √(Real.log (2 : ℝ) / Real.log (3 : ℝ)) := by
        have h17 : (Real.log (3 : ℝ) / Real.log (2 : ℝ)) + (Real.log (2 : ℝ) / Real.log (3 : ℝ)) + 2 ≥ 0 := by
          nlinarith [h14, h15]
        have h18 : Real.log (3 : ℝ) / Real.log (2 : ℝ) ≥ 0 := h14
        have h19 : Real.log (2 : ℝ) / Real.log (3 : ℝ) ≥ 0 := h15
        have h20 : (√(Real.log (3 : ℝ) / Real.log (2 : ℝ)) + √(Real.log (2 : ℝ) / Real.log (3 : ℝ))) ^ 2 = (Real.log (3 : ℝ) / Real.log (2 : ℝ)) + (Real.log (2 : ℝ) / Real.log (3 : ℝ)) + 2 := by
          have h23 : (√(Real.log (3 : ℝ) / Real.log (2 : ℝ)) + √(Real.log (2 : ℝ) / Real.log (3 : ℝ))) ^ 2 =
            (√(Real.log (3 : ℝ) / Real.log (2 : ℝ))) ^ 2 + (√(Real.log (2 : ℝ) / Real.log (3 : ℝ))) ^ 2 + 2 * (√(Real.log (3 : ℝ) / Real.log (2 : ℝ)) * √(Real.log (2 : ℝ) / Real.log (3 : ℝ))) := by
            ring
          have h24 : (√(Real.log (3 : ℝ) / Real.log (2 : ℝ))) ^ 2 = Real.log (3 : ℝ) / Real.log (2 : ℝ) := by
            apply Real.sq_sqrt
            linarith [h14]
          have h25 : (√(Real.log (2 : ℝ) / Real.log (3 : ℝ))) ^ 2 = Real.log (2 : ℝ) / Real.log (3 : ℝ) := by
            apply Real.sq_sqrt
            linarith [h15]
          have h26 : √(Real.log (3 : ℝ) / Real.log (2 : ℝ)) * √(Real.log (2 : ℝ) / Real.log (3 : ℝ)) = 1 := by
            calc
              √(Real.log (3 : ℝ) / Real.log (2 : ℝ)) * √(Real.log (2 : ℝ) / Real.log (3 : ℝ))
                = √((Real.log (3 : ℝ) / Real.log (2 : ℝ)) * (Real.log (2 : ℝ) / Real.log (3 : ℝ))) := by
                  rw [Real.sqrt_mul (by linarith [h14])]
              _ = √1 := by
                have h27 : (Real.log (3 : ℝ) / Real.log (2 : ℝ)) * (Real.log (2 : ℝ) / Real.log (3 : ℝ)) = 1 := by
                  field_simp [h4, h5]
                rw [h27]
              _ = 1 := Real.sqrt_one
          rw [h23, h24, h25, h26]
          all_goals linarith
        have h29 : √((Real.log (3 : ℝ) / Real.log (2 : ℝ)) + (Real.log (2 : ℝ) / Real.log (3 : ℝ)) + 2) ≥ 0 := Real.sqrt_nonneg _
        have h30 : √(Real.log (3 : ℝ) / Real.log (2 : ℝ)) + √(Real.log (2 : ℝ) / Real.log (3 : ℝ)) ≥ 0 := by
          apply add_nonneg
          · apply Real.sqrt_nonneg _
          · apply Real.sqrt_nonneg _
        have h31 : √((Real.log (3 : ℝ) / Real.log (2 : ℝ)) + (Real.log (2 : ℝ) / Real.log (3 : ℝ)) + 2) = √(Real.log (3 : ℝ) / Real.log (2 : ℝ)) + √(Real.log (2 : ℝ) / Real.log (3 : ℝ)) := by
          have h32 : (Real.log (3 : ℝ) / Real.log (2 : ℝ)) + (Real.log (2 : ℝ) / Real.log (3 : ℝ)) + 2 ≥ 0 := by
            nlinarith [h14, h15]
          have h33 : √((Real.log (3 : ℝ) / Real.log (2 : ℝ)) + (Real.log (2 : ℝ) / Real.log (3 : ℝ)) + 2) = √(Real.log (3 : ℝ) / Real.log (2 : ℝ)) + √(Real.log (2 : ℝ) / Real.log (3 : ℝ)) := by
            rw [←h20]
            field_simp [h29, h32, h18, h19]
          exact h33
        linarith [h31]
      rw [h13, h16]
    exact h8

