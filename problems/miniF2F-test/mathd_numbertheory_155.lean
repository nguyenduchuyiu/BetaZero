import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_155 :
  Finset.card (Finset.filter (fun x => x ≡ 7 [MOD 19]) (Finset.Icc (100:ℕ) 999)) = 48 := by
  sorry
