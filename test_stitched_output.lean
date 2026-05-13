
  have h_expr : ∀ x ∈ S, f x = 30 - x := by
    intro x hx
    rw [hS] at hx
    rw [h₀]
    dsimp
    have hxp : 0 ≤ x - p := by linarith [hx.1]
    have hx15 : x - 15 ≤ 0 := by linarith [hx.2]
    have hx_p_15 : x - p - 15 < 0 := by linarith [hx.2, h₁.1]
    rw [abs_of_nonneg hxp, abs_of_nonpos hx15, abs_of_neg hx_p_15]
    ring
  have h_lower_bound : ∀ y ∈ R, 15 ≤ y := by
    intro y hy
    rw [hR] at hy
    rcases hy with ⟨x, hxS, rfl⟩
    rw [h_expr x hxS]
    rw [hS] at hxS
    have hx_le : x ≤ 15 := hxS.2
    linarith
  have h_mem_R : 15 ∈ R := by
    rw [hR]
    use 15
    constructor
    · rw [hS]
      exact Set.right_mem_Icc.mpr (le_of_lt h₁.2)
    · rw [h_expr 15]
      · norm_num
      · rw [hS]
        exact Set.right_mem_Icc.mpr (le_of_lt h₁.2)
  have h_final : IsLeast R 15 := by
    constructor
    · exact h_mem_R
    · exact h_lower_bound
  exact h_final