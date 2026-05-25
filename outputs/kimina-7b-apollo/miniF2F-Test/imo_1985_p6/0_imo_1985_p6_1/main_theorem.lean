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
lemma imo_1985_p6_1
  (f : ℕ → NNReal → ℝ)
  (h₀ : ∀ (x : NNReal), f (0 : ℕ) x * f (0 : ℕ) x = (↑x : ℝ))
  (h₁ : ∀ (x : NNReal) (n : ℕ), f ((1 : ℕ) + n) x = f n x * (f n x + (↑n : ℝ)⁻¹)) :
  ∃! a : NNReal
    ∀ (n : ℕ)
      (0 : ℕ) < n →
        (0 : ℝ) < f n a ∧ f n a < f n a * (↑n : ℝ)⁻¹ + f n a ^ (2 : ℕ) ∧ f n a * (↑n : ℝ)⁻¹ + f n a ^ (2 : ℕ) < (1 : ℝ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry