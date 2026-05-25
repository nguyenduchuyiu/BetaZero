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
lemma mathd_numbertheory_483_1
  (a : ℕ → ℕ)
  (h₀ : a (1 : ℕ) = (1 : ℕ))
  (h₁ : a (2 : ℕ) = (1 : ℕ))
  (h₂ : ∀ (n : ℕ), a (n + (2 : ℕ)) = a (n + (1 : ℕ)) + a n) :
  a (100 : ℕ) % (4 : ℕ) = (3 : ℕ) := by
  have h100 : a 100 % 4 = 3 := by
    norm_num [h₀, h₁, h₂]
  exact h100