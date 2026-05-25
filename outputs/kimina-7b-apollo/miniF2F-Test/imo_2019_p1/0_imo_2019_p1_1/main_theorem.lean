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
lemma imo_2019_p1_1
  (f : ℤ → ℤ) :
  (∀ (a b : ℤ), f ((2 : ℤ) * a) + (2 : ℤ) * f b = f (f (a + b))) ↔
    ∀ (z : ℤ), f z = (0 : ℤ) ∨ ∃ (c : ℤ), ∀ (z : ℤ), f z = (2 : ℤ) * z + c := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry