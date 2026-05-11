import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem algebra_amgm_prod1toneq1_sum1tongeqn (a : ℕ → ℝ) (n : ℕ)
  (h₀: ∀ x:ℕ, 0 ≤ a x)
  (h₀ : Finset.prod (Finset.range n) a = 1) : Finset.sum (Finset.range n) a ≥ n := by
  sorry
