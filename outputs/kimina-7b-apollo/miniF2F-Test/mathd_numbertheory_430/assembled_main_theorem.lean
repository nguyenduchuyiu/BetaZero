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
theorem mathd_numbertheory_430
  (a b c : ℕ)
  (h₀ : 1 ≤ a ∧ a ≤ 9)
  (h₁ : 1 ≤ b ∧ b ≤ 9)
  (h₂ : 1 ≤ c ∧ c ≤ 9)
  (h₃ : a ≠ b)
  (h₄ : a ≠ c)
  (h₅ : b ≠ c)
  (h₆ : a + b = c)
  (h₇ : 10 * a + a - b = 2 * c)
  (h₈ : c * b = 10 * a + a + a) :
  a + b + c = 8 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
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

