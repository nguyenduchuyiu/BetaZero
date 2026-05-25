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
theorem mathd_numbertheory_135
  (n A B C : ℕ)
  (h₀ : n = 3^17 + 3^10)
  (h₁ : 11 ∣ (n + 1))
  (h₂ : [A,B,C].Pairwise (·≠·))
  (h₃ : {A,B,C} ⊂ Finset.Icc 0 9)
  (h₄ : Odd A ∧ Odd C)
  (h₅ : ¬ 3 ∣ B)
  (h₆ : Nat.digits 10 n = [B,A,B,C,C,A,C,B,A]) :
  100 * A + 10 * B + C = 129 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have hA : A = 1 := by linarith
    have hB : B = 2 := by linarith
    have hC : C = 9 := by linarith
    calc
      A * 100 + B * 10 + C = 1 * 100 + 2 * 10 + 9 := by rw [hA, hB, hC]
      _ = 100 + 20 + 9 := by norm_num
      _ = 129 := by norm_num

