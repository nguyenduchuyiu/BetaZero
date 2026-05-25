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
lemma algebra_apbon2pownleqapownpbpowon2_1
  (a b : ℝ)
  (n : ℕ)
  (h₀ : (0 : ℝ) < a ∧ (0 : ℝ) < b)
  (h₁ : (0 : ℕ) < n) :
  ((a + b) / (2 : ℝ)) ^ n ≤ (a ^ n + b ^ n) / (2 : ℝ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry