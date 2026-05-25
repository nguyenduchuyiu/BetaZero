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
theorem mathd_algebra_452
  (a : ℕ → ℝ)
  (h₀ : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)
  (h₁ : a 1 = 2 / 3)
  (h₂ : a 9 = 4 / 5) :
  a 5 = 11 / 15 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h₃ := h₀ 1
    have h₄ := h₀ 2
    have h₅ := h₀ 3
    have h₆ := h₀ 4
    have h₇ := h₀ 5
    have h₈ := h₀ 6
    have h₉ := h₀ 7
    have h₁₀ := h₀ 8
    norm_num at h₃ h₄ h₅ h₆ h₇ h₈ h₉ h₁₀
    linarith

