import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2021_p3
  (x y : ℕ)
  (h₀ : x + y = 17402)
  (h₁ : 10 ∣ x)
  (h₂ : (Nat.digits 10 x).drop 1 = Nat.digits 10 y)
  (h₂ : x / 10 = y) :
  abs ((↑x:ℤ) - ↑y) = (14238 : ℤ) := by
  sorry
