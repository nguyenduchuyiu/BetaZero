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
lemma imo_1992_p1_1
  (p q r : ℤ)
  (h₀ : (1 : ℤ) < p ∧ p < q ∧ q < r)
  (h₁ : (p - (1 : ℤ)) * (q - (1 : ℤ)) * (r - (1 : ℤ)) ∣ p * q * r - (1 : ℤ)) :
  p = (2 : ℤ) ∧ q = (4 : ℤ) ∧ r = (8 : ℤ) ∨ p = (3 : ℤ) ∧ q = (5 : ℤ) ∧ r = (15 : ℤ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry