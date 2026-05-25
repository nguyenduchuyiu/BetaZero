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
theorem aime_1988_p8
  (f : ℕ → ℕ → ℝ)
  (h₀ : ∀ x, 0 < x → f x x = x)
  (h₁ : ∀ x y, (0 < x ∧ 0 < y) → f x y = f y x)
  (h₂ : ∀ x y, (0 < x ∧ 0 < y) → (↑x + ↑y) * f x y = y * (f x (x + y))) :
  f 14 52 = 364 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    
    have h1 := h₂ 14 52 (by norm_num) (by norm_num)
    have h2 := h₂ 14 38 (by norm_num) (by norm_num)
    have h3 := h₂ 14 24 (by norm_num) (by norm_num)
    have h4 := h₂ 14 10 (by norm_num) (by norm_num)
    have h5 := h₂ 10 14 (by norm_num) (by norm_num)
    have h6 := h₂ 10 4 (by norm_num) (by norm_num)
    have h7 := h₂ 4 10 (by norm_num) (by norm_num)
    have h8 := h₂ 4 6 (by norm_num) (by norm_num)
    have h9 := h₂ 4 2 (by norm_num) (by norm_num)
    have h10 := h₂ 2 4 (by norm_num) (by norm_num)
    have h11 := h₂ 2 2 (by norm_num) (by norm_num)
    have h12 := h₀ 2 (by norm_num)
    
    norm_num [h₁, h₀] at *
    
    linarith

