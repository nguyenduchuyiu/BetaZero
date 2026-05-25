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
lemma algebra_cubrtrp1oncubrtreq3_rcubp1onrcubeq5778_1
  (r : ℝ)
  (h₀ : r ^ (1 / 3 : ℝ) + (r ^ (1 / 3 : ℝ))⁻¹ = (3 : ℝ)) :
  r ^ (3 : ℕ) + r⁻¹ ^ (3 : ℕ) = (5778 : ℝ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry