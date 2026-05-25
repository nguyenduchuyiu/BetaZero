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
theorem mathd_algebra_158
  (a : ℕ)
  (h₀ : Even a)
  (h₁ : ∑ k ∈ Finset.range 8, (2 * k + 1) - ∑ k ∈ Finset.range 5, (a + 2 * k) = (4:ℤ)) :
  a = 8 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h2 : ∑ x in Finset.range 5, ((↑a : ℤ) + (↑x : ℤ) * (2 : ℤ)) = 5 * (a : ℤ) + 20 := by
      simp only [Finset.sum_range_succ, add_mul, mul_add]
      ring
    rw [h2] at h₁
    have h3 : (a : ℤ) = 8 := by
      linarith
    exact_mod_cast h3

