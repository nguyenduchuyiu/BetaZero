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
lemma amc12a_2020_p15_1
  (a b : ℂ)
  (h₀ : a ^ (3 : ℕ) - (8 : ℂ) = (0 : ℂ))
  (h₁ : b ^ (3 : ℕ) - (8 : ℂ) * b ^ (2 : ℕ) - (8 : ℂ) * b + (64 : ℂ) = (0 : ℂ)) :
  ‖a - b‖ ≤ (2 : ℝ) * √(21 : ℝ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry