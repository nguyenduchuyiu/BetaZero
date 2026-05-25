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
lemma induction_1pxpownlt1pnx_1
  (x : ℝ)
  (n : ℕ)
  (h₀ : (-1 : ℝ) < x)
  (h₁ : (0 : ℕ) < n) :
  (1 : ℝ) + (↑n : ℝ) * x ≤ ((1 : ℝ) + x) ^ n := by
  induction n with
  | zero => 
    cases h₁
  | succ n ih =>
    cases n with
    | zero =>
      simp
    | succ n =>
      simp [pow_succ] at ih ⊢
      nlinarith [sq_nonneg (x * (↑n : ℝ)), sq_nonneg (x), h₀]