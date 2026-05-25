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
lemma amc12a_2003_p5_1
  (A M C : ℕ)
  (h₀ : A ≤ (9 : ℕ) ∧ M ≤ (9 : ℕ) ∧ C ≤ (9 : ℕ))
  (h₁ : ofDigits (10 : ℕ) [(0 : ℕ), (1 : ℕ), C, M, A] + ofDigits (10 : ℕ) [(2 : ℕ), (1 : ℕ), C, M, A] = (123422 : ℕ)) :
  A + M + C = (14 : ℕ) := by
    simp [ofDigits, List.foldl] at h₁
    omega