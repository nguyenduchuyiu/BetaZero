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
theorem amc12a_2020_p7
  (a : ℕ → ℕ)
  (h₀ : (a 0)^3 = 1)
  (h₁ : (a 1)^3 = 8)
  (h₂ : (a 2)^3 = 27)
  (h₃ : (a 3)^3 = 64)
  (h₄ : (a 4)^3 = 125)
  (h₅ : (a 5)^3 = 216)
  (h₆ : (a 6)^3 = 343) :


  
  have ha0 : a 0 = 1 := by
    have h : a 0 ^ 3 = 1 := by simpa using h₀
    have : a 0 = 1 := by
      have h1 : a 0 ^ 3 = 1 := h
      have h2 : a 0 ≤ 1 := by
        by_contra h2
        push_neg at h2
        have h3 : a 0 ^ 3 > 1 := by
          have : a 0 ≥ 2 := by omega
          have h4 : a 0 ^ 3 ≥ 2 ^ 3 := by
            apply Nat.pow_le_pow_of_le_left this 3
          norm_num at h4
          linarith
        linarith
      interval_cases a 0 <;> tauto
    exact this
  have ha1 : a 1 = 2 := by
    have h : a 1 ^ 3 = 8 := by simpa using h₁
    have : a 1 = 2 := by
      have h1 : a 1 ^ 3 = 8 := h
      have h2 : a 1 ≤ 2 := by
        by_contra h2
        push_neg at h2
        have h3 : a 1 ^ 3 > 8 := by
          have : a 1 ≥ 3 := by omega
          have h4 : a 1 ^ 3 ≥ 3 ^ 3 := by
            apply Nat.pow_le_pow_of_le_left this 3
          norm_num at h4
          linarith
        linarith
      interval_cases a 1 <;> tauto
    exact this
  have ha2 : a 2 = 3 := by
    have h : a 2 ^ 3 = 27 := by simpa using h₂
    have : a 2 = 3 := by
      have h1 : a 2 ^ 3 = 27 := h
      have h2 : a 2 ≤ 3 := by
        by_contra h2
        push_neg at h2
        have h3 : a 2 ^ 3 > 27 := by
          have : a 2 ≥ 4 := by omega
          have h4 : a 2 ^ 3 ≥ 4 ^ 3 := by
            apply Nat.pow_le_pow_of_le_left this 3
          norm_num at h4
          linarith
        linarith
      interval_cases a 2 <;> tauto
    exact this
  have ha3 : a 3 = 4 := by
    have h : a 3 ^ 3 = 64 := by simpa using h₃
    have : a 3 = 4 := by
      have h1 : a 3 ^ 3 = 64 := h
      have h2 : a 3 ≤ 4 := by
        by_contra h2
        push_neg at h2
        have h3 : a 3 ^ 3 > 64 := by
          have : a 3 ≥ 5 := by omega
          have h4 : a 3 ^ 3 ≥ 5 ^ 3 := by
            apply Nat.pow_le_pow_of_le_left this 3
          norm_num at h4
          linarith
        linarith
      interval_cases a 3 <;> tauto
    exact this
  have ha4 : a 4 = 5 := by
    have h : a 4 ^ 3 = 125 := by simpa using h₄
    have : a 4 = 5 := by
      have h1 : a 4 ^ 3 = 125 := h
      have h2 : a 4 ≤ 5 := by
        by_contra h2
        push_neg at h2
        have h3 : a 4 ^ 3 > 125 := by
          have : a 4 ≥ 6 := by omega
          have h4 : a 4 ^ 3 ≥ 6 ^ 3 := by
            apply Nat.pow_le_pow_of_le_left this 3
          norm_num at h4
          linarith
        linarith
      interval_cases a 4 <;> tauto
    exact this
  have ha5 : a 5 = 6 := by
    have h : a 5 ^ 3 = 216 := by simpa using h₅
    have : a 5 = 6 := by
      have h1 : a 5 ^ 3 = 216 := h
      have h2 : a 5 ≤ 6 := by
        by_contra h2
        push_neg at h2
        have h3 : a 5 ^ 3 > 216 := by
          have : a 5 ≥ 7 := by omega
          have h4 : a 5 ^ 3 ≥ 7 ^ 3 := by
            apply Nat.pow_le_pow_of_le_left this 3
          norm_num at h4
          linarith
        linarith
      interval_cases a 5 <;> tauto
    exact this
  have ha6 : a 6 = 7 := by
    have h : a 6 ^ 3 = 343 := by simpa using h₆
    have : a 6 = 7 := by
      have h1 : a 6 ^ 3 = 343 := h
      have h2 : a 6 ≤ 7 := by
        by_contra h2
        push_neg at h2
        have h3 : a 6 ^ 3 > 343 := by
          have : a 6 ≥ 8 := by omega
          have h4 : a 6 ^ 3 ≥ 8 ^ 3 := by
            apply Nat.pow_le_pow_of_le_left this 3
          norm_num at h4
          linarith
        linarith
      interval_cases a 6 <;> tauto
    exact this
  have hsum1 : ∑ k in Finset.range 7, (6 : ℕ) * a k ^ 2 = 6 * (1 ^ 2 + 2 ^ 2 + 3 ^ 2 + 4 ^ 2 + 5 ^ 2 + 6 ^ 2 + 7 ^ 2) := by
    simp [Finset.sum_range_succ, ha0, ha1, ha2, ha3, ha4, ha5, ha6]
  have hsum2 : ∑ k in Finset.range 6, a k ^ 2 = 1 ^ 2 + 2 ^ 2 + 3 ^ 2 + 4 ^ 2 + 5 ^ 2 + 6 ^ 2 := by
    simp [Finset.sum_range_succ, ha0, ha1, ha2, ha3, ha4, ha5]
  calc
    ∑ k ∈ Finset.range (7 : ℕ), (6 : ℕ) * a k ^ (2 : ℕ) - (2 : ℕ) * ∑ k ∈ Finset.range (6 : ℕ), a k ^ (2 : ℕ)
        = 6 * (1 ^ 2 + 2 ^ 2 + 3 ^ 2 + 4 ^ 2 + 5 ^ 2 + 6 ^ 2 + 7 ^ 2) - (2 : ℕ) * (1 ^ 2 + 2 ^ 2 + 3 ^ 2 + 4 ^ 2 + 5 ^ 2 + 6 ^ 2) := by rw [hsum1, hsum2]
    _ = 6 * (1 + 4 + 9 + 16 + 25 + 36 + 49) - 2 * (1 + 4 + 9 + 16 + 25 + 36) := by norm_num
    _ = 6 * 140 - 2 * 91 := by norm_num
    _ = 840 - 182 := by norm_num
    _ = 658 := by norm_num

