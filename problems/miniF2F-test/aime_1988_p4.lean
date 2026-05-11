import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem aime_1988_p4
  (S : Set ℕ+)
  (a : ℕ+ → ℝ)
  (hS : S = {n | (∀ i:ℕ+, i ∈ Finset.Icc 1 n → abs (a i) < 1) → (∑ k ∈ Finset.Icc 1 n, abs (a k)) = 19 + abs (∑ k ∈ Finset.Icc 1 n, a k)}) :
  IsLeast S 20 := by
  sorry
