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
lemma aime_1984_p1_1
  (u : ℕ → ℚ)
  (h₀ : ∀ (n : ℕ), u (n + (1 : ℕ)) = u n + (1 : ℚ))
  (h₁ : ∑ x ∈ Finset.range (98 : ℕ), (u x + (1 : ℚ)) = (137 : ℚ)) :
  ∑ x ∈ Finset.range (49 : ℕ), u ((2 : ℕ) + x * (2 : ℕ)) = (93 : ℚ) := by
  have h2 : ∀ n, u (n + 1) = u n + 1 := by
    intro n
    exact h₀ n
  norm_num [Finset.sum_range_succ, h2] at h₁ ⊢
  linarith