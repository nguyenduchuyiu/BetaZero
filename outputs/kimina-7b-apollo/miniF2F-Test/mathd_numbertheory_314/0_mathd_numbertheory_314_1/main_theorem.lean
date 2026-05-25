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
lemma mathd_numbertheory_314_1
  (r n : ℕ)
  (h₀ : r = (3 : ℕ))
  (h₁ : (0 : ℕ) < n)
  (h₂ : (1342 : ℕ) ∣ n)
  (h₃ : n % (13 : ℕ) < (3 : ℕ)) :
  (6710 : ℕ) ≤ n := by
    by_contra h
    push_neg at h
    have h4 : n % 13 < 3 := h₃
    have h5 : 1342 ∣ n := h₂
    have h6 : n < 6710 := h
    have h7 : n ≤ 6710 := by linarith
    interval_cases n <;> norm_num at *
    <;> omega