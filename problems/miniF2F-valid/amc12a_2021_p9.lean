import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2021_p9 :
  (∏ k ∈ Finset.range (7:ℕ), ((2: ℕ)^(2^k: ℕ) + (3: ℕ)^(2^(k: ℕ)))) = (3: ℕ)^(128: ℕ) - (2: ℕ)^(128: ℕ) := by
  sorry
