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
lemma mathd_numbertheory_764_1
  (p : ℕ)
  (h₀ : Nat.Prime p)
  (h₁ : (7 : ℕ) ≤ p) :
  ∑ x ∈ Finset.Icc (1 : ℕ) (p - (2 : ℕ)), (↑x : ZMod p)⁻¹ * ((↑x : ZMod p) + (1 : ZMod p))⁻¹ = (2 : ZMod p) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry