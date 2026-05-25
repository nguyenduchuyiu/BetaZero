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
lemma mathd_numbertheory_451_1
  (S : Finset ℕ)
  (h₀ : ∀ (n : ℕ), n ∈ S ↔ (2010 : ℕ) ≤ n ∧ n ≤ (2019 : ℕ) ∧ ∃ (m : ℕ), m.divisors.card = (4 : ℕ) ∧ ∑ p ∈ m.divisors, p = n) :
  ∑ k ∈ S, k = (2016 : ℕ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry