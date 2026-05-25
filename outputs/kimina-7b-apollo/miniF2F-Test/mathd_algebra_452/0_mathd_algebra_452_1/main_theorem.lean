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
lemma mathd_algebra_452_1
  (a : ℕ → ℝ)
  (h₀ : ∀ (n : ℕ), a (n + (2 : ℕ)) - a (n + (1 : ℕ)) = a (n + (1 : ℕ)) - a n)
  (h₁ : a (1 : ℕ) = (2 / 3 : ℝ))
  (h₂ : a (9 : ℕ) = (4 / 5 : ℝ)) :
  a (5 : ℕ) = (11 / 15 : ℝ) := by
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