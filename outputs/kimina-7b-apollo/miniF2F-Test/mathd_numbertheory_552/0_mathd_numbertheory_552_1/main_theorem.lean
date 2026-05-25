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
lemma mathd_numbertheory_552_1
  (f g h : ℕ+ → ℕ)
  (h₀ : ∀ (x : ℕ+), f x = (12 : ℕ) * (↑x : ℕ) + (7 : ℕ))
  (h₁ : ∀ (x : ℕ+), g x = (5 : ℕ) * (↑x : ℕ) + (2 : ℕ))
  (h₃ : Fintype (↑(Set.range h) : Type))
  (h₂ : ∀ (x : ℕ+), h x = ((7 : ℕ) + (↑x : ℕ) * (12 : ℕ)).gcd ((2 : ℕ) + (↑x : ℕ) * (5 : ℕ))) :
  ∑ k ∈ (Set.range h).toFinset, k = (12 : ℕ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry