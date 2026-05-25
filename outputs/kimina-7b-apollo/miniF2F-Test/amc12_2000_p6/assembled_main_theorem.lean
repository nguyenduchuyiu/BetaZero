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
theorem amc12_2000_p6
  (p q : ℕ)
  (h₀ : Nat.Prime p ∧ Nat.Prime q)
  (h₁ : 4 ≤ p ∧ p ≤ 18)
  (h₂ : 4 ≤ q ∧ q ≤ 18) :
  p * q - (p + q) ≠ 194 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have hp : p ≤ 18 := h₁.2
    have hq : q ≤ 18 := h₂.2
    interval_cases p <;> interval_cases q
    <;> norm_num at *
    <;> try { contradiction }

