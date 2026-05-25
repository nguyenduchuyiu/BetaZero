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
lemma mathd_numbertheory_457_1
  (n : ℕ)
  (h₀ : (0 : ℕ) < n)
  (h₁ : (80325 : ℕ) ∣ n !) :
  (17 : ℕ) ≤ n := by
    by_contra h
    push_neg at h
    interval_cases n <;> contradiction