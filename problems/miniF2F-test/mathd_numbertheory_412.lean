import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_412 (x y : ℤ) (h₀ : x ≡ 4 [ZMOD 19]) (h₁ : y ≡7 [ZMOD 19]) :
  ((x + 1) ^ 2 * (y + 5) ^ 3) % 19 = 13 := by
  sorry
