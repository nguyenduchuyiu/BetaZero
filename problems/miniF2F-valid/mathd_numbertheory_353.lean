import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_353 (s : ℕ) (h₀ : s = ∑ k ∈Finset.Icc 2010 4018, k) : s % 2009 = 0 := by
  sorry
