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
lemma imo_2001_p6_1
    (a b c d : ℕ)
    (h₀ : (0 : ℕ) < a ∧ (0 : ℕ) < b ∧ (0 : ℕ) < c ∧ (0 : ℕ) < d)
    (h₁ : d < c)
    (h₂ : c < b)
    (h₃ : b < a)
    (h₄ : a * c + b * d = (b + d + a - c) * (b + d + c - a)) :
    ¬Nat.Prime (a * b + c * d) := by
  sorry