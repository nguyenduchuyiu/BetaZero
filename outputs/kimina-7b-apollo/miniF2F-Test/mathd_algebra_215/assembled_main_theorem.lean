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
theorem mathd_algebra_215
  (S : Finset ℝ)
  (h₀ : ∀ (x : ℝ), x ∈ S ↔ (x + 3)^2 = 121) :
  ∑ k ∈ S, k = -6 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h1 : S = {8, -14} := by
      ext x
      simp [h₀]
      constructor
      · -- Show that if x ∈ S, then x ∈ {8, -14}
        intro h
        have h_eq : (x + 3) ^ 2 = 121 := by
          simpa using h
        have h2 : x + 3 = 11 ∨ x + 3 = -11 := by
          have h3 : (x + 3) ^ 2 - 121 = 0 := by linarith
          have h4 : ((x + 3) - 11) * ((x + 3) + 11) = 0 := by
            ring_nf at h3 ⊢
            linarith
          cases (mul_eq_zero.mp h4) with
          | inl h5 => 
            left
            linarith
          | inr h6 => 
            right
            linarith
        cases h2 with
        | inl h3 =>
          left
          linarith
        | inr h3 =>
          right
          linarith
      · -- Show that if x ∈ {8, -14}, then x ∈ S
        rintro (h | h)
        · -- x = 8
          rw [h]
          norm_num
        · -- x = -14
          rw [h]
          norm_num
    rw [h1]
    norm_num [Finset.sum_insert, Finset.sum_singleton]

