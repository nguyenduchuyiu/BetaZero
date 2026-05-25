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
lemma amc12a_2021_p22_1
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h₁ : f ⁻¹' {(0 : ℝ)} = {cos (π * (2 / 7 : ℝ)), cos (π * (4 / 7 : ℝ)), cos (π * (6 / 7 : ℝ))})
  (h₀ : ∀ (x : ℝ), f x = a * x ^ (2 : ℕ) + b * x + c + x ^ (3 : ℕ)) :
  a * b * c = (1 / 32 : ℝ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry