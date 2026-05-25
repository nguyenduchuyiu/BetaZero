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
theorem mathd_algebra_196
  (S : Finset ℝ)
  (h₀ : ∀ (x : ℝ), x ∈ S ↔ abs (2 - x) = 3) :
  ∑ k ∈ S, k = 4 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h1 : S = {-1, 5} := by
      ext x
      simp [h₀]
      constructor
      · -- Assume |2 - x| = 3, prove x ∈ {-1, 5}
        intro h
        cases eq_or_eq_neg_of_abs_eq h with
        | inl h1 =>
          left
          linarith
        | inr h2 =>
          right
          linarith
      · -- Assume x ∈ {-1, 5}, prove |2 - x| = 3
        rintro (h | h)
        · -- x = -1
          rw [h]
          norm_num
        · -- x = 5
          rw [h]
          norm_num
    rw [h1]
    rw [Finset.sum_insert]
    rw [Finset.sum_singleton]
    all_goals norm_num

