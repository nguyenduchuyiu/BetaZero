import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_427 (a : ℕ) (h₀ : a = ∑ k ∈Nat.divisors 500, k) :
  (∑ k ∈ Finset.filter (fun x => Nat.Prime x) (Nat.divisors a), k) = 25 := by
  sorry
