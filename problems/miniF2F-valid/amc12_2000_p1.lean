import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12_2000_p1:   IsGreatest {z : ℕ | ∃ I M O : ℕ, I > 0 ∧ M > 0 ∧O > 0 ∧ I ≠ M ∧ M ≠ O ∧ I * M * O = 2001 ∧ I + M + O = z} 671 := by
  sorry
