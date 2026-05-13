import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12_2001_p5 :
  Finset.prod (Finset.filter (fun (x : ℕ) => Odd x) (Finset.range 10000)) (fun (x : ℕ) => x) =
  10000! / (2 ^ (5000 : ℕ) * 5000!) := by
  sorry
