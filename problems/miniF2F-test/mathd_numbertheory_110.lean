import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_110 (a b : ℕ) (h₀ : b ≤ a) (h₁ : (a + b) ≡ 2 [MOD 10])
  (h₂ : (2 * a + b) ≡ 1 [MOD 10]) : (a - b) % 10 = 6 := by
  sorry
