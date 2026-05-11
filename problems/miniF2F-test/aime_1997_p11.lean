import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem aime_1997_p11 (x : ℝ)
  (h₀ : x =
  (∑ n ∈ Finset.Icc (1 : ℕ) 44, Real.cos (n * Real.pi / 180)) /
  ∑ n ∈ Finset.Icc (1 : ℕ) 44, Real.sin (n * Real.pi / 180)) :
  Int.floor (100 * x) = 241 := by
  sorry
