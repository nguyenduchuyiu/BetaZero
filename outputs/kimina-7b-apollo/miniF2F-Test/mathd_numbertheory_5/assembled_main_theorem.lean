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
theorem mathd_numbertheory_5
  (n : ℕ)
  (h₀ : 10 ≤ n)
  (h₁ : ∃ x, x^2 = n)
  (h₂ : ∃ t, t^3 = n) :
  64 ≤ n := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    rcases h₁ with ⟨x, hx⟩
    rcases h₂ with ⟨t, ht⟩
    by_contra h
    push_neg at h
    have h3 : x ≤ 7 := by
      by_contra hx'
      push_neg at hx'
      have h4 : x ^ 2 ≥ 8 ^ 2 := by
        have h5 : x ≥ 8 := by linarith
        exact Nat.pow_le_pow_of_le_left h5 2
      have h6 : x ^ 2 ≥ 64 := by
        norm_num at h4 ⊢
        linarith
      have h7 : n ≥ 64 := by linarith [hx.symm, h6]
      linarith
    have h4 : t ≤ 3 := by
      by_contra ht'
      push_neg at ht'
      have h5 : t ^ 3 ≥ 4 ^ 3 := by
        have h6 : t ≥ 4 := by linarith
        exact Nat.pow_le_pow_of_le_left h6 3
      have h7 : t ^ 3 ≥ 64 := by
        norm_num at h5 ⊢
        linarith
      have h8 : n ≥ 64 := by linarith [ht.symm, h7]
      linarith
    interval_cases x <;> 
    interval_cases t <;> 
    omega

