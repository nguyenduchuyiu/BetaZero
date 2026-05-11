import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_343
  (S : Finset ℕ)
  (hS : S = Finset.filter (fun x => Odd x) (Finset.range 13))
  (p : ℕ)
  (h₀: p = ∏ k ∈ S, k) :
  p % 10 = 5 := by
  sorry
