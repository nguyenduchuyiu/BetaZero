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
lemma mathd_numbertheory_222_1
  (b : ℕ)
  (h₀ : Nat.lcm (120 : ℕ) b = (3720 : ℕ))
  (h₁ : Nat.gcd (120 : ℕ) b = (8 : ℕ)) :
  b = (248 : ℕ) := by
  have h2 : 120 * b = 8 * 3720 := by
    calc
      120 * b = Nat.gcd (120 : ℕ) b * Nat.lcm (120 : ℕ) b := by rw [Nat.gcd_mul_lcm]
      _ = 8 * 3720 := by rw [h₀, h₁]
  have h3 : b = 248 := by
    omega
  exact h3