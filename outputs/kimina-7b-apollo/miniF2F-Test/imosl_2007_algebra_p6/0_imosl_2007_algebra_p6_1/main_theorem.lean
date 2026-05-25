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
lemma imosl_2007_algebra_p6_1
  (a : ℕ → NNReal)
  (h₀ : ∑ x ∈ Finset.range (100 : ℕ), a (x + (1 : ℕ)) ^ (2 : ℕ) = (1 : NNReal)) :
  ∑ x ∈ Finset.range (99 : ℕ), a (x + (1 : ℕ)) ^ (2 : ℕ) * a (x + (2 : ℕ)) + a (100 : ℕ) ^ (2 : ℕ) * a (1 : ℕ) <
    (12 : NNReal) / (25 : NNReal) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry