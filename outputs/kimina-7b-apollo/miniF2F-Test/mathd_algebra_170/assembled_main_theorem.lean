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
theorem mathd_algebra_170
  (S : Finset ℤ)
  (h₀ : ∀ (n : ℤ), n ∈ S ↔ abs (n - 2) ≤ 5 + 6 / 10) :
  S.card = 11 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h1 : S = Finset.Icc (-3) 7 := by
      ext n
      simp [h₀]
      constructor
      · -- If |(-2 : ℤ) + n| ≤ (5 : ℤ), then -3 ≤ n ≤ 7
        intro h
        rw [abs_le] at h
        have h2 : -5 ≤ (-2 : ℤ) + n := h.1
        have h3 : (-2 : ℤ) + n ≤ 5 := h.2
        constructor <;> linarith
      · -- If -3 ≤ n ≤ 7, then |(-2 : ℤ) + n| ≤ (5 : ℤ)
        rintro ⟨h1, h2⟩
        rw [abs_le]
        constructor
        · linarith
        · linarith
    rw [h1]
    native_decide

