import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_421 (a b c d : ℝ)
  (Sf Sg : Set (ℝ × ℝ))
  (h₀: ∀ x y:ℝ, (x, y) ∈ Sf ↔ y = x ^ 2 + 4 * x + 6)
  (h₁: ∀ x y:ℝ, (x, y) ∈ Sg ↔ y = 1 / 2 * x ^ 2 + x + 6)
  (h₂: (a, b) ∈ Sf ∩ Sg)
  (h₃: (c, d) ∈ Sf ∩ Sg)
  (h₄ : (a, b) ≠ (c, d))
  (h₅ : a ≤ c) :
  c - a = 6 := by
  sorry
