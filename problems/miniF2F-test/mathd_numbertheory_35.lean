import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_35
  (S : Finset ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ n ∣ Nat.sqrt 196)
  (h₁: S.card = 4) :
  (∑ k ∈ S, k) = 24 := by
  sorry
