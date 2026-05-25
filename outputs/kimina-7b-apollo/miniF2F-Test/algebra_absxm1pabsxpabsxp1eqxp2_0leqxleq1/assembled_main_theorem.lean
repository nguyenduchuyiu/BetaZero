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
theorem algebra_absxm1pabsxpabsxp1eqxp2_0leqxleq1
  (x : ℝ)
  (h₀ : abs (x - 1) + abs x + abs (x + 1) = x + 2) :
  0 ≤ x ∧ x ≤ 1 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    by_cases h1 : x ≥ 1
    · -- Case 1: x ≥ 1
      rw [abs_of_nonneg (by linarith), abs_of_nonneg (by linarith), abs_of_nonneg (by linarith)] at h₀
      constructor
      · linarith
      · linarith
    · -- Case 2: x < 1
      by_cases h2 : x ≥ 0
      · -- Subcase 2.1: 0 ≤ x < 1
        rw [abs_of_nonpos (by linarith), abs_of_nonneg (by linarith), abs_of_nonneg (by linarith)] at h₀
        constructor
        · linarith
        · linarith
      · -- Case 2.2: x < 0
        by_cases h3 : x ≥ -1
        · -- Subcase 2.2.1: -1 ≤ x < 0
          rw [abs_of_nonpos (by linarith), abs_of_nonpos (by linarith), abs_of_nonneg (by linarith)] at h₀
          constructor
          · linarith
          · linarith
        · -- Case 2.2.2: x < -1
          rw [abs_of_nonpos (by linarith), abs_of_nonpos (by linarith), abs_of_nonpos (by linarith)] at h₀
          constructor
          · linarith
          · linarith

