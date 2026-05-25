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
lemma amc12a_2002_p6_1
  (n : ℕ)
  (h₀ : (0 : ℕ) < n) :
  ∃ (m : ℕ), n < m ∧ ∃ (p : ℕ), m * p ≤ m + p := by
    use (n + 1)
    constructor
    · -- Show that n < n + 1
      omega
    · -- Show that ∃ p, (n + 1) * p ≤ (n + 1) + p
      use 1
      omega