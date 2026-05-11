import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2010_p22 : IsLeast {x | ∃ t : ℝ, x = ∑ i in Finset.range 119, |(i+1)*t - 1|} 49 := by
  sorry
