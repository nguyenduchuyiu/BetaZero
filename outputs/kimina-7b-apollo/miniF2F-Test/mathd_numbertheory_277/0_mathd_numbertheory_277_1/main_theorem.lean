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
lemma mathd_numbertheory_277_1
  (m n : ℕ)
  (h₀ : m.gcd n = (6 : ℕ))
  (h₁ : m.lcm n = (126 : ℕ)) :
  (60 : ℕ) ≤ m + n := by
  have h2 : m * n = 6 * 126 := by
    calc
      m * n = m.gcd n * m.lcm n := by rw [Nat.gcd_mul_lcm]
      _ = 6 * 126 := by rw [h₀, h₁]
  by_contra h
  push_neg at h
  have h3 : m ≤ 60 := by nlinarith
  have h4 : n ≤ 60 := by nlinarith
  interval_cases m <;> interval_cases n
  <;> norm_num at h2 h₀ h₁
  <;> all_goals
    contradiction