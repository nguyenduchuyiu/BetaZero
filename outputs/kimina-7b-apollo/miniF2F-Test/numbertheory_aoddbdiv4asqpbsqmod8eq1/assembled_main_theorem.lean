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
theorem numbertheory_aoddbdiv4asqpbsqmod8eq1
  (a : ℤ)
  (b : ℤ)
  (h₀ : Odd a)
  (h₁ : 4 ∣ b)
  (h₂ : b >= 0) :
  (a^2 + b^2) % 8 = 1 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h1 : a ^ 2 % 8 = 1 := by
      obtain ⟨k, hk⟩ := h₀
      rw [hk]
      ring_nf
      have : (4 * (k ^ 2 + k) : ℤ) % 8 = 0 := by
        have h : (k ^ 2 + k : ℤ) % 2 = 0 := by
          have h1 : (k % 2 = 0) ∨ (k % 2 = 1) := by omega
          rcases h1 with (h1 | h1)
          · simp [h1, pow_two, Int.add_emod, Int.mul_emod]
          · simp [h1, pow_two, Int.add_emod, Int.mul_emod]
        omega
      omega
    have h2 : b ^ 2 % 8 = 0 := by
      obtain ⟨m, hm⟩ := h₁
      rw [hm]
      ring_nf
      omega
    have h3 : (a ^ 2 + b ^ 2) % 8 = 1 := by
      omega
    exact h3

