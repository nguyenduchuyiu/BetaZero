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
lemma numbertheory_notEquiv2i2jasqbsqdiv8_1:
  ∃ (x : ℤ) (x_1 : ℤ)
    ¬(((∃ (x_2 : ℤ), x = (2 : ℤ) * x_2) ∧ ∃ (x : ℤ), x_1 = (2 : ℤ) * x) ↔
        ∃ (k : ℤ), x ^ (2 : ℕ) + x_1 ^ (2 : ℕ) = (8 : ℤ) * k) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry