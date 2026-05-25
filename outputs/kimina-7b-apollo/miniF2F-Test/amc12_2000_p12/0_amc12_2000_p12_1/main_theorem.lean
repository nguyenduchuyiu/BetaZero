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
lemma amc12_2000_p12_1
  (a m c : ℕ)
  (h₀ : a + m + c = (12 : ℕ)) :
  a * m * c + a * m + m * c + a * c ≤ (112 : ℕ) := by
  have h1 : a ≤ 12 := by linarith
  have h2 : m ≤ 12 := by linarith
  have h3 : c ≤ 12 := by linarith
  interval_cases a <;> interval_cases m <;> omega