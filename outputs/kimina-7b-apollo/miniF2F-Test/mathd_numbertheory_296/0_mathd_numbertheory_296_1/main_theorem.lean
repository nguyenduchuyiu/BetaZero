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
lemma mathd_numbertheory_296_1
  (n : ℕ)
  (h₀ : (2 : ℕ) ≤ n)
  (h₁ : ∃ (x : ℕ), x ^ (3 : ℕ) = n)
  (h₂ : ∃ (t : ℕ), t ^ (4 : ℕ) = n) :
  (4096 : ℕ) ≤ n := by 
  rcases h₁ with ⟨x, hx⟩
  rcases h₂ with ⟨t, ht⟩
  by_contra h
  push_neg at h
  have h3 : x ≤ 16 := by
    by_contra hx'
    push_neg at hx'
    have h4 : x ^ 3 ≥ 17 ^ 3 := by
      have h5 : x ≥ 17 := by linarith
      exact Nat.pow_le_pow_of_le_left h5 3
    norm_num at h4
    linarith
  have h4 : t ≤ 7 := by
    by_contra ht'
    push_neg at ht'
    have h5 : t ^ 4 ≥ 8 ^ 4 := by
      have h6 : t ≥ 8 := by linarith
      exact Nat.pow_le_pow_of_le_left h6 4
    norm_num at h5
    linarith
  interval_cases t <;> 
  interval_cases x <;> 
  omega