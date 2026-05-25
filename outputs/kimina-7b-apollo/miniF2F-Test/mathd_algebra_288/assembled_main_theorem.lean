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
theorem mathd_algebra_288
  (x y : ℝ)
  (n : NNReal)
  (h₀ : x < 0 ∧ y < 0)
  (h₁ : abs y = 6)
  (h₂ : Real.sqrt ((x - 8)^2 + (y - 3)^2) = 15)
  (h₃ : Real.sqrt (x^2 + y^2) = Real.sqrt n) :
  n = 52 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have hy : y = -6 := by
      have h₁' : y = -6 := by
        cases abs_cases y with
        | inl h => linarith
        | inr h => linarith
      exact h₁'
    have h4 : (x - 8) ^ 2 + (y - 3) ^ 2 = (15 : ℝ) ^ 2 := by
      calc
        (x - 8) ^ 2 + (y - 3) ^ 2 = (√((x - 8) ^ 2 + (y - 3) ^ 2)) ^ 2 := by
          rw [Real.sq_sqrt]
          positivity
        _ = (15 : ℝ) ^ 2 := by rw [h₂]
    rw [hy] at h4
    have hx : x = -4 := by
      nlinarith [h₀.left, hy]
    have h5 : x ^ 2 + y ^ 2 = (↑n : ℝ) := by
      calc
        x ^ 2 + y ^ 2 = (√(x ^ 2 + y ^ 2)) ^ 2 := by
          rw [Real.sq_sqrt]
          positivity
        _ = (√(↑n : ℝ)) ^ 2 := by rw [h₃]
        _ = (↑n : ℝ) := by
          rw [Real.sq_sqrt]
          positivity
    rw [hx, hy] at h5
    have hn : (n : ℝ) = 52 := by
      norm_num at h5
      linarith
    exact_mod_cast hn

