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
lemma mathd_numbertheory_618_1
  (n : ℕ)
  (p : ℕ → ℕ)
  (h₁ : (1 : ℕ) < ((41 : ℕ) + (n ^ (2 : ℕ) - n)).gcd ((41 : ℕ) + ((1 : ℕ) + n * (2 : ℕ) + n ^ (2 : ℕ) - ((1 : ℕ) + n))))
  (h₀ : ∀ (x : ℕ), p x = (41 : ℕ) + (x ^ (2 : ℕ) - x)) :
  (41 : ℕ) ≤ n := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry