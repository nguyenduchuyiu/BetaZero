import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12_2001_p2
  (n : ℕ)
  (P S : ℕ → ℕ)
  (h₀ : P = fun x => (Nat.digits 10 x).prod)
  (h₁ : S = fun x => (Nat.digits 10 x).sum)
  (h₂ : (Nat.digits 10 n).length = 2)
  (h₃ : n = P n + S n) :
  n % 10 = 9 := by
  sorry
