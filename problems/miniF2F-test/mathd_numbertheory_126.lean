import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem mathd_numbertheory_126
  (x a : ℤ)
  (h₀ : 0 < x)
  (h₁: x = 40 ∨ a = 40)
  (Sa Sx : Set ℤ)
  (hSa : Sa = {b | Int.gcd b x = x + 3 ∧ Int.lcm b x = x * (x + 3)})
  (hSx : Sx = {y | 0 < y ∧ Int.gcd a y = y + 3 ∧ Int.lcm a y = y * (y + 3)})
  (h₂ : Nonempty Sa)
  (h₃ : Nonempty Sx) :
  IsLeast Sa 8 ∨ IsLeast Sx 8 := by
  sorry
