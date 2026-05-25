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
lemma amc12b_2021_p4_1
  (m a : ℕ)
  (h₀ : (0 : ℕ) < m ∧ (0 : ℕ) < a)
  (h₁ : (↑m : ℝ) / (↑a : ℝ) = (3 / 4 : ℝ)) :
  ((84 : ℝ) * (↑m : ℝ) + (70 : ℝ) * (↑a : ℝ)) / ((↑m : ℝ) + (↑a : ℝ)) = (76 : ℝ) := by
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