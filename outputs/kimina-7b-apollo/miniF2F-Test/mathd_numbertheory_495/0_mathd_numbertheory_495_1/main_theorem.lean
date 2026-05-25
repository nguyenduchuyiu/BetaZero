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
lemma mathd_numbertheory_495_1
  (a b : ℕ)
  (h₀ : (0 : ℕ) < a ∧ (0 : ℕ) < b)
  (h₁ : a % (10 : ℕ) = (2 : ℕ))
  (h₂ : b % (10 : ℕ) = (4 : ℕ))
  (h₃ : a.gcd b = (6 : ℕ)) :
  (108 : ℕ) ≤ a.lcm b := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry