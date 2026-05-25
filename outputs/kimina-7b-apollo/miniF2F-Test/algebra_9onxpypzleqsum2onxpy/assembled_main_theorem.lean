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
theorem algebra_9onxpypzleqsum2onxpy
  (x y z : ℝ)
  (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) :
  9 / (x + y + z) ≤ 2 / (x + y) + 2 / (y + z) + 2 / (z + x) := by
    try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith


    
    have h1 : 0 < x := h₀.1
    have h2 : 0 < y := h₀.2.1
    have h3 : 0 < z := h₀.2.2
    have h4 : 0 < x + y := by linarith
    have h5 : 0 < y + z := by linarith
    have h6 : 0 < z + x := by linarith
    have h7 : 0 < x + y + z := by linarith
    have h8 : (2 / (x + y) + 2 / (y + z) + 2 / (z + x)) ≥ 9 / (x + y + z) := by
      have h9 : (2 / (x + y) + 2 / (y + z) + 2 / (z + x)) - 9 / (x + y + z) ≥ 0 := by
        have h10 : (2 / (x + y) + 2 / (y + z) + 2 / (z + x)) - 9 / (x + y + z) =
          (2 * (y + z) * (z + x) * (x + y + z) + 2 * (z + x) * (x + y) * (x + y + z) + 2 * (x + y) * (y + z) * (x + y + z) - 9 * (x + y) * (y + z) * (z + x)) / ((x + y) * (y + z) * (z + x) * (x + y + z)) := by
          field_simp
          ring
        rw [h10]
        have h11 : 2 * (y + z) * (z + x) * (x + y + z) + 2 * (z + x) * (x + y) * (x + y + z) + 2 * (x + y) * (y + z) * (x + y + z) - 9 * (x + y) * (y + z) * (z + x) ≥ 0 := by
          nlinarith [sq_nonneg (x - y), sq_nonneg (y - z), sq_nonneg (z - x),
            sq_nonneg (x + y), sq_nonneg (y + z), sq_nonneg (z + x),
            sq_nonneg (x + y + z)]
        have h12 : (x + y) * (y + z) * (z + x) * (x + y + z) > 0 := by
          apply mul_pos
          apply mul_pos
          all_goals nlinarith
        apply div_nonneg
        · linarith
        · nlinarith
      linarith
    linarith

