import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_24
  (x : ℕ)
  (h₀ : x = ∑ k ∈ Finset.Icc 1 9, 11 ^ k)
  (L : List ℕ)
  (h₁ : L = (Nat.digits 10 x).drop 1)
  (h₂ : L ≠ []) :
  List.head L h₂ = 5 := by
  sorry
