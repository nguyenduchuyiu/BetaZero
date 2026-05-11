import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem numbertheory_aoddbdiv4asqpbsqmod8eq1 
  (a : ℤ) (b : ℕ) (h₀ : Odd a) (h₁ : 4 ∣ b) : 
  (a ^ 2 + b ^ 2) ≡ 1 [ZMOD 8] := by
  sorry
