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
lemma mathd_numbertheory_135_1
  (n A B C : ℕ)
  (h₀ : n = (129199212 : ℕ))
  (h₁ : (11 : ℕ) ∣ (129199213 : ℕ))
  (h₂ : (¬A = B ∧ ¬A = C) ∧ ¬B = C)
  (h₃ : {A, B, C} ⊂ Finset.Icc (0 : ℕ) C)
  (h₄ : Odd A ∧ Odd C)
  (h₅ : ¬(3 : ℕ) ∣ B)
  (h₆ : (2 : ℕ) = B ∧ (1 : ℕ) = A ∧ (2 : ℕ) = B ∧ (9 : ℕ) = C ∧ (1 : ℕ) = A ∧ (9 : ℕ) = C ∧ (2 : ℕ) = B ∧ (1 : ℕ) = A) :
  A * (100 : ℕ) + B * (10 : ℕ) + C = (129 : ℕ) := by
  have hA : A = 1 := by linarith
  have hB : B = 2 := by linarith
  have hC : C = 9 := by linarith
  calc
    A * 100 + B * 10 + C = 1 * 100 + 2 * 10 + 9 := by rw [hA, hB, hC]
    _ = 100 + 20 + 9 := by norm_num
    _ = 129 := by norm_num