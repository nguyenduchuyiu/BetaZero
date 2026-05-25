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
lemma mathd_algebra_342_1
  (a d : ℝ)
  (h₀ : ∑ k ∈ Finset.range (5 : ℕ), (a + (↑k : ℝ) * d) = (70 : ℝ))
  (h₁ : ∑ k ∈ Finset.range (10 : ℕ), (a + (↑k : ℝ) * d) = (210 : ℝ)) :
  a = (42 / 5 : ℝ) := by
    have eq1 : ∑ k in Finset.range 5, (a + (↑k : ℝ) * d) = 5 * a + 10 * d := by
      simp [Finset.sum_range_succ, add_mul, mul_add]
      ring
    have eq2 : ∑ k in Finset.range 10, (a + (↑k : ℝ) * d) = 10 * a + 45 * d := by
      simp [Finset.sum_range_succ, add_mul, mul_add]
      ring
    rw [eq1] at h₀
    rw [eq2] at h₁
    have h2 : 5 * a + 10 * d = 70 := by linarith
    have h3 : 10 * a + 45 * d = 210 := by linarith
    have h4 : a = 42 / 5 := by
      linarith
    exact h4