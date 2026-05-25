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
lemma mathd_algebra_320_1
  (x : ℝ)
  (a b c : ℕ)
  (h₃ : c = (2 : ℕ))
  (h₂ : x = (↑a : ℝ) * (1 / 2 : ℝ) + √(↑b : ℝ) * (1 / 2 : ℝ))
  (h₁ :
  (↑a : ℝ) * √(↑b : ℝ) + (↑a : ℝ) ^ (2 : ℕ) * (1 / 2 : ℝ) + √(↑b : ℝ) ^ (2 : ℕ) * (1 / 2 : ℝ) =
    (9 : ℝ) + (↑a : ℝ) * (2 : ℝ) + √(↑b : ℝ) * (2 : ℝ))
  (h₀ : (0 : ℕ) < a ∧ (0 : ℕ) < b ∧ (0 : ℝ) ≤ (↑a : ℝ) * (1 / 2 : ℝ) + √(↑b : ℝ) * (1 / 2 : ℝ)) :
  a + b = (24 : ℕ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry