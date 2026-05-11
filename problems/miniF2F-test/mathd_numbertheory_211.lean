import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_211 :
  Finset.card ((Finset.filter (fun n => (4 * n ≡ 2 [MOD 6]))) (Finset.Ioo (0:ℕ) 60)) = 20 := by
  sorry
