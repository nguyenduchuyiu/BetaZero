import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_32 (S : Finset ℕ) (h₀ : ∀ n : ℕ, n ∈ S ↔ (0 < n ∧ n ∣ 36)) : (∑ k ∈ S, k) = 91 := by
  sorry
