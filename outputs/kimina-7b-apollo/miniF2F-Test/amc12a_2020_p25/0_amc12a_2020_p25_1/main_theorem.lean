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
lemma amc12a_2020_p25_1
  (a : ℚ)
  (S : Finset ℝ)
  (h₀ : ∀ (x : ℝ), x ∈ S ↔ (↑⌊x⌋ : ℝ) * Int.fract x = (↑a : ℝ) * x ^ (2 : ℕ))
  (h₁ : ∑ k ∈ S, k = (420 : ℝ)) :
  (↑a.den : ℤ) + a.num = (929 : ℤ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry