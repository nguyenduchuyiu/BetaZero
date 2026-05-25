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
lemma amc12_2001_p5_1:
    (Finset.filter (fun (x : ℕ) => ¬Even x) (Finset.range (10000 : ℕ))).prod id =
        (10000 : ℕ)! / ((2 : ℕ) ^ (5000 : ℕ) * (5000 : ℕ)!) := by
  sorry