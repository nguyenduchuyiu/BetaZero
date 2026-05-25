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
lemma aime_1990_p4_1
  (x : ℝ)
  (h₀ : (0 : ℝ) < x)
  (h₄ :
  ((-29 : ℝ) - x * (10 : ℝ) + x ^ (2 : ℕ))⁻¹ +
      (((-45 : ℝ) - x * (10 : ℝ) + x ^ (2 : ℕ))⁻¹ - ((-69 : ℝ) - x * (10 : ℝ) + x ^ (2 : ℕ))⁻¹ * (2 : ℝ)) =
    (0 : ℝ))
  (h₃ : ¬(-69 : ℝ) - x * (10 : ℝ) + x ^ (2 : ℕ) = (0 : ℝ))
  (h₂ : ¬(-45 : ℝ) - x * (10 : ℝ) + x ^ (2 : ℕ) = (0 : ℝ))
  (h₁ : ¬(-29 : ℝ) - x * (10 : ℝ) + x ^ (2 : ℕ) = (0 : ℝ)) :
  x = (13 : ℝ) := by
    have h5 : x = (13 : ℝ) := by
      have h6 : (-29 : ℝ) - x * (10 : ℝ) + x ^ (2 : ℕ) ≠ 0 := by
        intro h
        apply h₁
        linarith
      have h7 : (-45 : ℝ) - x * (10 : ℝ) + x ^ (2 : ℕ) ≠ 0 := by
        intro h
        apply h₂
        linarith
      have h8 : (-69 : ℝ) - x * (10 : ℝ) + x ^ (2 : ℕ) ≠ 0 := by
        intro h
        apply h₃
        linarith
      field_simp at h₄
      nlinarith [sq_nonneg (x ^ 2 - 10 * x + 27), sq_nonneg (x - 13), sq_nonneg (x - 3), sq_nonneg (x - 11), sq_nonneg (x - 9), sq_nonneg (x - 7), sq_nonneg (x - 5), sq_nonneg (x - 1)]
    exact h5