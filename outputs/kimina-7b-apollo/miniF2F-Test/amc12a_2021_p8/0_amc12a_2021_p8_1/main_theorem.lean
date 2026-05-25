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
lemma amc12a_2021_p8_1
    (d : ℕ → ℕ)
    (h₀ : d (0 : ℕ) = (0 : ℕ))
    (h₁ : d (1 : ℕ) = (0 : ℕ))
    (h₂ : d (2 : ℕ) = (1 : ℕ))
    (h₃ : ∀ n ≥ (3 : ℕ), d n = d (n - (1 : ℕ)) + d (n - (3 : ℕ))) :
    Even (d (2021 : ℕ)) ∧ Odd (d (2022 : ℕ)) ∧ Even (d (2023 : ℕ)) := by
  sorry