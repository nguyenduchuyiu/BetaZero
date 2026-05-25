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
lemma mathd_numbertheory_234_1
  (a b : ℕ)
  (h₀ : (1 : ℕ) ≤ a ∧ a ≤ (9 : ℕ) ∧ b ≤ (9 : ℕ))
  (h₁ : ((10 : ℕ) * a + b) ^ (3 : ℕ) = (912673 : ℕ)) :
  a + b = (16 : ℕ) := by
    have h2 : 10 * a + b = 97 := by
      have h3 : (10 * a + b) ^ 3 = 912673 := h₁
      have h4 : 10 * a + b = 97 := by
        have h5 : (10 * a + b) ^ 3 = 912673 := h₁
        have h6 : 10 * a + b ≤ 99 := by
          have h7 : a ≤ 9 := h₀.right.left
          have h8 : b ≤ 9 := h₀.right.right
          omega
        have h9 : 10 * a + b ≥ 10 := by
          have h10 : 1 ≤ a := h₀.left
          have h11 : 0 ≤ b := by linarith
          omega
        interval_cases (10 * a + b) <;> norm_num at h5 <;> omega
      exact h4
    have ha : a = 9 := by
      omega
    have hb : b = 7 := by
      omega
    calc
      a + b = 9 + 7 := by rw [ha, hb]
      _ = 16 := by norm_num