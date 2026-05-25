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
lemma mathd_algebra_289_1
  (k t m n : ℕ)
  (h₀ : Nat.Prime m ∧ Nat.Prime (0 : ℕ))
  (h₁ : t < k)
  (h₂ : k ^ (2 : ℕ) - m * k = (0 : ℕ) ∧ n = (0 : ℕ))
  (h₃ : t ^ (2 : ℕ) - m * t = (0 : ℕ)) :
  (1 : ℕ) + k ^ t + t ^ k + (0 : ℕ) ^ m = (20 : ℕ) := by
    have h0_false : ¬ Nat.Prime (0 : ℕ) := by
      exact Nat.not_prime_zero
    have h_contra := h₀.right
    contradiction