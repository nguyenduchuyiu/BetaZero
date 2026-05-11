import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem numbertheory_3pow2pownm1mod2pownp3eq2pownp2
  (n : ℕ)
  (h₀ : 0 < n) :
  (3 ^ 2 ^ (n : ℕ) - 1) ≡ 2 ^ ((n : ℕ) + 2) [MOD 2 ^ ((n : ℕ) + 3)] := by
  sorry
