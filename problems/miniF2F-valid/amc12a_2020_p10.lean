import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12a_2020_p10
  (n : ℕ)
  (h₀ : 1 < n)
  (h₁ : Real.logb (2 : ℝ) (Real.logb (16 : ℝ) n) = Real.logb (4 : ℝ) (Real.logb (4 : ℝ) n)) :
  List.sum (Nat.digits 10 n) = 13 := by
  sorry
