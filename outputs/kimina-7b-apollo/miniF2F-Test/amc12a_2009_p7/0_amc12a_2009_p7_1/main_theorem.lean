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
lemma amc12a_2009_p7_1
  (x : ℝ)
  (n : ℕ)
  (a : ℕ → ℝ)
  (h₁ : ∀ (m : ℕ), a (m + (1 : ℕ)) - a m = a (m + (2 : ℕ)) - a (m + (1 : ℕ)))
  (h₂ : a (1 : ℕ) = (2 : ℝ) * x - (3 : ℝ))
  (h₃ : a (2 : ℕ) = (5 : ℝ) * x - (11 : ℝ))
  (h₄ : a (3 : ℕ) = (3 : ℝ) * x + (1 : ℝ))
  (h₅ : a n = (2009 : ℝ)) :
  n = (502 : ℕ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry