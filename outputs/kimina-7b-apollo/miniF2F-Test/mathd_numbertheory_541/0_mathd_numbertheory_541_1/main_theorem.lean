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
lemma mathd_numbertheory_541_1
  (m n : ℕ)
  (h₀ : (1 : ℕ) < m)
  (h₁ : (1 : ℕ) < n)
  (h₂ : m * n = (2005 : ℕ)) :
  m + n = (406 : ℕ) := by
  have h3 : m ≤ 2005 := by nlinarith
  have h4 : n ≤ 2005 := by nlinarith
  interval_cases m <;> omega