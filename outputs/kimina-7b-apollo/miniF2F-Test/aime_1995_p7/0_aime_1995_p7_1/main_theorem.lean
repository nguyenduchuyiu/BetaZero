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
lemma aime_1995_p7_1
  (k m n : ℕ)
  (t : ℝ)
  (h₀ : (0 : ℕ) < k ∧ (0 : ℕ) < m ∧ (0 : ℕ) < n)
  (h₁ : m.gcd n = (1 : ℕ))
  (h₂ : ((1 : ℝ) + sin t) * ((1 : ℝ) + cos t) = (5 / 4 : ℝ))
  (h₃ : ((1 : ℝ) - sin t) * ((1 : ℝ) - cos t) = (↑m : ℝ) / (↑n : ℝ) - √(↑k : ℝ)) :
  k + m + n = (27 : ℕ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry