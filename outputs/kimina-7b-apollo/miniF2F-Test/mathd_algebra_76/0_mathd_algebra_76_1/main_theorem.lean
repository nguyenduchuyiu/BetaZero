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
lemma mathd_algebra_76_1
  (f : ℤ → ℤ)
  (h₀ : ∀ (n : ℤ), Odd n → f n = n ^ (2 : ℕ))
  (h₁ : ∀ (n : ℤ), Even n → f n = (-1 : ℤ) - n * (4 : ℤ) + n ^ (2 : ℕ)) :
  f (4 : ℤ) = (-1 : ℤ) := by
  have h4 : f (4 : ℤ) = (-1 : ℤ) - (4 : ℤ) * (4 : ℤ) + (4 : ℤ) ^ (2 : ℕ) := by
    apply h₁
    exact ⟨2, by norm_num⟩
  rw [h4]
  norm_num