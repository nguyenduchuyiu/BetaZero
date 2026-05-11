import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_160
  (n x : ℕ)
  (f : ℕ → ℕ)
  (h₀ : f = fun t => n + x * t)
  (h₁: f 1 = 97)
  (h₂: f 5 = 265) :
  f 2 = 139 := by
  sorry
