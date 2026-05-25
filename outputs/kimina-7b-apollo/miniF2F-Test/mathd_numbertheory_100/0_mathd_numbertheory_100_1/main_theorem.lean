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
lemma mathd_numbertheory_100_1
  (n : ℕ)
  (h₀ : (0 : ℕ) < n)
  (h₁ : n.gcd (40 : ℕ) = (10 : ℕ))
  (h₂ : n.lcm (40 : ℕ) = (280 : ℕ)) :
  n = (70 : ℕ) := by
  have h3 : n.gcd 40 * n.lcm 40 = n * 40 := by
    rw [Nat.gcd_mul_lcm]
  rw [h₁, h₂] at h3
  have h4 : 10 * 280 = n * 40 := h3
  have h5 : n = 70 := by
    omega
  exact h5