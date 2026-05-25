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
lemma mathd_numbertheory_521_1
  (m n : ℕ)
  (h₀ : Even m)
  (h₁ : Even n)
  (h₂ : m - n = (2 : ℕ))
  (h₃ : m * n = (288 : ℕ)) :
  m = (18 : ℕ) := by
    have h4 : m = n + 2 := by
      omega
    rw [h4] at h₃
    have : n ≤ 17 := by nlinarith
    interval_cases n <;> omega