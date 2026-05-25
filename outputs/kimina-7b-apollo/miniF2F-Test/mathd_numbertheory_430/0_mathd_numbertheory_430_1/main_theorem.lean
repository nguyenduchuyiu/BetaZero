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
lemma mathd_numbertheory_430_1
  (a b c : ℕ)
  (h₀ : (1 : ℕ) ≤ a ∧ a ≤ (9 : ℕ))
  (h₁ : (1 : ℕ) ≤ b ∧ b ≤ (9 : ℕ))
  (h₂ : (1 : ℕ) ≤ c ∧ c ≤ (9 : ℕ))
  (h₃ : ¬a = b)
  (h₄ : ¬a = c)
  (h₅ : ¬b = c)
  (h₈ : c * b = a * (12 : ℕ))
  (h₇ : a * (11 : ℕ) - b = c * (2 : ℕ))
  (h₆ : b + a = c) :
  c * (2 : ℕ) = (8 : ℕ) := by
  have h9 : b = 3 * a := by
    omega
  have h10 : c = 4 * a := by
    omega
  have h11 : a = 1 := by
    nlinarith [h₀.left, h₀.right, h9, h10, h₈]
  have h12 : c = 4 := by
    omega
  calc
    c * (2 : ℕ) = 4 * (2 : ℕ) := by rw [h12]
    _ = 8 := by norm_num