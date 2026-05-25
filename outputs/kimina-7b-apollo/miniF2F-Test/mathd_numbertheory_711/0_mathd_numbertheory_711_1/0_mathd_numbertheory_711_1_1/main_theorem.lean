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
lemma mathd_numbertheory_711_1_1
  (m n : ℕ)
  (h₀ : (0 : ℕ) < m ∧ (0 : ℕ) < n)
  (h₁ : m.gcd n = (8 : ℕ))
  (h₂ : m.lcm n = (112 : ℕ)) :
  (72 : ℕ) ≤ m + n := by
    have h3 : m.gcd n * m.lcm n = m * n := by
      rw [Nat.gcd_mul_lcm]
    rw [h₁, h₂] at h3
    have h4 : m * n = 896 := by
      linarith
    by_contra h
    push_neg at h
    have h5 : m < 72 := by
      nlinarith
    have h6 : n < 72 := by
      nlinarith
    interval_cases m <;> interval_cases n
    <;> norm_num at h₁ h₂ h4
    <;> all_goals
      contradiction