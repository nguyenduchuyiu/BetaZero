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
lemma amc12a_2020_p10_1
  (n : ℕ)
  (h₀ : (0 : ℕ) < n)
  (h₁ : logb (2 : ℝ) (logb (16 : ℝ) (↑n : ℝ)) = logb (4 : ℝ) (logb (4 : ℝ) (↑n : ℝ))) :
  n % (10 : ℕ) + (digits (10 : ℕ) (n / (10 : ℕ))).sum = (13 : ℕ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry