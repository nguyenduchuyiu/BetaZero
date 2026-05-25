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
lemma aime_1997_p9_1
  (a : ℝ)
  (h₀ : (0 : ℝ) < a)
  (h₁ : Int.fract a⁻¹ = Int.fract (a ^ (2 : ℕ)))
  (h₂ : (2 : ℝ) < a ^ (2 : ℕ))
  (h₃ : a ^ (2 : ℕ) < (3 : ℝ)) :
  a ^ (12 : ℕ) - a⁻¹ * (144 : ℝ) = (233 : ℝ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry