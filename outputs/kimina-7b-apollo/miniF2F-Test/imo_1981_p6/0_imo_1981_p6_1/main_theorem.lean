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
lemma imo_1981_p6_1
  (f : ℕ → ℕ → ℕ)
  (g : ℕ → ℕ)
  (h₀ : ∀ (y : ℕ), f (0 : ℕ) y = y + (1 : ℕ))
  (h₁ : ∀ (x : ℕ), f (x + (1 : ℕ)) (0 : ℕ) = f x (1 : ℕ))
  (h₂ : ∀ (x y : ℕ), f (x + (1 : ℕ)) (y + (1 : ℕ)) = f x (f (x + (1 : ℕ)) y))
  (h₃ : g (0 : ℕ) = (2 : ℕ))
  (h₄ : ∀ (n : ℕ), g (n + (1 : ℕ)) = (2 : ℕ) ^ g n) :
  f (4 : ℕ) (1981 : ℕ) = g (1983 : ℕ) - (3 : ℕ) := by
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith
  sorry