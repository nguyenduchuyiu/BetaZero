import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12b_2020_p5
  (na nb wa wb la lb: ℕ)
  (h₀ : wa = (2:ℚ) / 3 * na)
  (h₁ : wb = (5 : ℚ) / 8 * nb)
  (h₂ : wb = wa + 7)
  (h₃ : la = na - wa)
  (h₄ : lb = nb - wb)
  (h₅ : lb = la + 7) :
  na = 42 := by
  sorry
