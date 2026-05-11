import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem induction_pord1p1on2powklt5on2
  (n : ℕ) (h₀ : 0 < n) :
  (∏ k ∈ Finset.Icc (1:ℕ) n, ((1:ℝ) + (1 : ℝ) / 2 ^ k)) < (5 / 2 : ℝ) := by
  sorry
