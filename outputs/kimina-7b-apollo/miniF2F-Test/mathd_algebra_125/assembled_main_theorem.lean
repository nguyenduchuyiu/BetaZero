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
theorem mathd_algebra_125
  (x y : ℕ)
  (h₀ : 0 < x ∧ 0 < y)
  (h₁ : 5 * x = y)
  (h₂ : (↑x - (3:ℤ)) + (y - (3:ℤ)) = 30) :
  x = 6 := by linarith