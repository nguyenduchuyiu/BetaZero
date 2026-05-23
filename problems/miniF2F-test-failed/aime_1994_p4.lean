import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem aime_1994_p4 (n : ℕ) (h₀ : 0 < n)
  (h₀ : (∑ k ∈ Finset.Icc 1 n, Int.floor (Real.logb 2 k)) = 1994) : n = 312 := by
  sorry
