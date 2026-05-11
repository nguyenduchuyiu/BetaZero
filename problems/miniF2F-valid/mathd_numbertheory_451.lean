import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_451 (S : Finset ℕ)
  (h₀ :
  ∀ n : ℕ,
  n ∈ S ↔
  2010 ≤ n ∧ n ≤ 2019 ∧ ∃ m, (Nat.divisors m).card = 4 ∧ (∑ p ∈ Nat.divisors m, p) = n) :
  (∑ k ∈ S, k) = 2016 := by
  sorry
