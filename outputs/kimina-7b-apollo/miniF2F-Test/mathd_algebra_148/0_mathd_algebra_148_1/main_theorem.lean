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
lemma mathd_algebra_148_1
  (c : ℝ)
  (f : ℝ → ℝ)
  (h₁ : (-15 : ℝ) + c * (8 : ℝ) = (9 : ℝ))
  (h₀ : ∀ (x : ℝ), f x = (3 : ℝ) + (c * x ^ (3 : ℕ) - x * (9 : ℝ))) :
  c = (3 : ℝ) := by 
  linarith