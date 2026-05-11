import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_algebra_31 (x : ℝ) (u : ℕ → ℝ) (h₀ : ∀ n, u (n + 1) = Real.sqrt (x + u n))
  (h₁ : Filter.Tendsto u Filter.atTop (nhds 9)) : x = 72 := by
  sorry
