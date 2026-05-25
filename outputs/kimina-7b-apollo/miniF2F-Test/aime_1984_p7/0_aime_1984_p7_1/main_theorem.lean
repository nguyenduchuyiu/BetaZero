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
lemma aime_1984_p7_1
  (f : ℤ → ℤ)
  (h₀ : ∀ (n : ℤ), (1000 : ℤ) ≤ n → f n = n - (3 : ℤ))
  (h₁ : ∀ n < (1000 : ℤ), f n = f (f (n + (5 : ℤ)))) :
  f (84 : ℤ) = (997 : ℤ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry