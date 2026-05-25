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
lemma mathd_algebra_246_1
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h₂ : (2 : ℝ) + (a * (81 : ℝ) - b * (9 : ℝ)) = (2 : ℝ))
  (h₀ : ∀ (x : ℝ), f x = (5 : ℝ) + (a * x ^ (4 : ℕ) - b * x ^ (2 : ℕ)) + x) :
  (8 : ℝ) + (a * (81 : ℝ) - b * (9 : ℝ)) = (8 : ℝ) := by
    have h1 : a * (81 : ℝ) - b * (9 : ℝ) = 0 := by linarith
    have h3 : b = 9 * a := by
      linarith
    rw [h3]
    linarith