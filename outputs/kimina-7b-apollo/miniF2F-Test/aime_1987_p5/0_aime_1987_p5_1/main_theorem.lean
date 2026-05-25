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
lemma aime_1987_p5_1
  (x y : ℤ)
  (h₀ : y ^ (2 : ℕ) + (3 : ℤ) * (x ^ (2 : ℕ) * y ^ (2 : ℕ)) = (30 : ℤ) * x ^ (2 : ℕ) + (517 : ℤ)) :
  (3 : ℤ) * (x ^ (2 : ℕ) * y ^ (2 : ℕ)) = (588 : ℤ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry