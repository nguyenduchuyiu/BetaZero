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
theorem amc12b_2021_p4
  (m a : ℕ)
  (h₀ : 0 < m ∧ 0 < a)
  (h₁ : ↑m / ↑a = (3:ℝ) / 4) :
  (84 * ↑m + 70 * ↑a) / (↑m + ↑a) = (76:ℝ) := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h2 : (↑a : ℝ) ≠ 0 := by
      have ha : (↑a : ℝ) > (0 : ℝ) := by
        exact_mod_cast h₀.right
      linarith
    have h3 : (↑m : ℝ) ≠ 0 := by
      have hm : (↑m : ℝ) > (0 : ℝ) := by
        exact_mod_cast h₀.left
      linarith
    have h4 : (↑m : ℝ) = (3 / 4 : ℝ) * (↑a : ℝ) := by
      field_simp [h2] at h₁ ⊢
      linarith
    have h5 : ((84 : ℝ) * (↑m : ℝ) + (70 : ℝ) * (↑a : ℝ)) / ((↑m : ℝ) + (↑a : ℝ)) = (76 : ℝ) := by
      rw [h4]
      field_simp [h3, h2]
      ring
    exact h5

