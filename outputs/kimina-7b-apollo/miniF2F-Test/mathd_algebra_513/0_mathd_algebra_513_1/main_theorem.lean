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
lemma mathd_algebra_513_1
  (a b : ℝ)
  (h₀ : (3 : ℝ) * a + (2 : ℝ) * b = (5 : ℝ))
  (h₁ : a + b = (2 : ℝ)) :
  a = (1 : ℝ) ∧ b = (1 : ℝ) := by
  have hb : b = 1 := by
    linarith
  have ha : a = 1 := by
    linarith [h₁, hb]
  exact ⟨ha, hb⟩