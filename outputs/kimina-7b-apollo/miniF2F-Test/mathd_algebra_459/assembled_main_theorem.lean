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
theorem mathd_algebra_459
  (a b c d : ℚ)
  (h₀ : 3 * a = b + c + d)
  (h₁ : 4 * b = a + c + d)
  (h₂ : 2 * c = a + b + d)
  (h₃ : 8 * a + 10 * b + 6 * c = 24) :
  ↑d.den + d.num = 28 := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have ha : a = (5 / 4 : ℚ) * b := by
      linarith [h₀, h₁]
    have hc : c = (5 / 3 : ℚ) * b := by
      linarith [h₁, h₂]
    have hd : d = (13 / 12 : ℚ) * b := by
      linarith [h₀, ha, hc]
    have hb : b = (4 / 5 : ℚ) := by
      have h₄ : (8 : ℚ) * a + (10 : ℚ) * b + (6 : ℚ) * c = (24 : ℚ) := h₃
      rw [ha, hc] at h₄
      linarith
    have hd' : d = (13 / 15 : ℚ) := by
      rw [hb] at hd
      linarith
    have h_den : (↑d.den : ℤ) = 15 := by
      rw [hd']
      native_decide
    have h_num : (↑d.num : ℤ) = 13 := by
      rw [hd']
      native_decide
    rw [h_den, h_num]
    norm_num

