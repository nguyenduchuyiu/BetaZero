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
lemma algebra_apbmpcneq0_aeq0anbeq0anceq0_1
  (a b c : ℚ)
  (m n : ℝ)
  (h₀ : (0 : ℝ) < m ∧ (0 : ℝ) < n)
  (h₁ : m ^ (3 : ℕ) = (2 : ℝ))
  (h₂ : n ^ (3 : ℕ) = (4 : ℝ))
  (h₃ : (↑a : ℝ) + (↑b : ℝ) * m + (↑c : ℝ) * n = (0 : ℝ)) :
  a = (0 : ℚ) ∧ b = (0 : ℚ) ∧ c = (0 : ℚ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry