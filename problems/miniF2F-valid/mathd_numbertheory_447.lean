import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_447 :
  (∑ k ∈ Finset.filter (fun x => 3 ∣ x) (Finset.Icc (0:ℕ) 50), k % 10) = 78 := by
  sorry
