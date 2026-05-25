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
lemma imo_1974_p3_1
    (n : ℕ) :
    ¬(5 : ℕ) ∣
        ∑ k ∈ Finset.range (n + (1 : ℕ)), ((2 : ℕ) * n + (1 : ℕ)).choose ((2 : ℕ) * k + (1 : ℕ)) * (2 : ℕ) ^ ((3 : ℕ) * k) := by
  sorry