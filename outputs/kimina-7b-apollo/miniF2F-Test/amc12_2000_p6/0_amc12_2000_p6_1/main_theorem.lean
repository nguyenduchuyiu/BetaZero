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
lemma amc12_2000_p6_1
  (p q : ℕ)
  (h₀ : Nat.Prime p ∧ Nat.Prime q)
  (h₁ : (4 : ℕ) ≤ p ∧ p ≤ (18 : ℕ))
  (h₂ : (4 : ℕ) ≤ q ∧ q ≤ (18 : ℕ)) :
  ¬p * q - (p + q) = (194 : ℕ) := by
    have hp : p ≤ 18 := h₁.2
    have hq : q ≤ 18 := h₂.2
    interval_cases p <;> interval_cases q
    <;> norm_num at *
    <;> try { contradiction }