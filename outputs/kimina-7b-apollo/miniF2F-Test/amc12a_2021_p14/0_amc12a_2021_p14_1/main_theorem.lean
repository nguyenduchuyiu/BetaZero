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
lemma amc12a_2021_p14_1:
  (∑ x ∈ Finset.Icc (1 : ℕ) (20 : ℕ), logb ((5 : ℝ) ^ x) ((3 : ℝ) ^ x ^ (2 : ℕ))) *
      ∑ x ∈ Finset.Icc (1 : ℕ) (100 : ℕ), logb ((9 : ℝ) ^ x) ((25 : ℝ) ^ x) =
    (21000 : ℝ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry