import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Rat Finset Topology

theorem amc12_2000_p6 :
  ¬ ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ 4 ≤ p ∧ p ≤ 18 ∧ 4 ≤ p ∧ p ≤ 18 ∧ p ≠ q ∧ ↑p * ↑q - (↑p + ↑q) = 22 ∧
  ¬ ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ 4 ≤ p ∧ p ≤ 18 ∧ 4 ≤ p ∧ p ≤ 18 ∧ p ≠ q ∧ ↑p * ↑q - (↑p + ↑q) = 60 ∧
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ 4 ≤ p ∧ p ≤ 18 ∧ 4 ≤ p ∧ p ≤ 18 ∧ p ≠ q ∧ ↑p * ↑q - (↑p + ↑q) = 119 ∧
  ¬ ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ 4 ≤ p ∧ p ≤ 18 ∧ 4 ≤ p ∧ p ≤ 18 ∧ p ≠ q ∧ ↑p * ↑q - (↑p + ↑q) = 180 ∧
  ¬ ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ 4 ≤ p ∧ p ≤ 18 ∧ 4 ≤ p ∧ p ≤ 18 ∧ p ≠ q ∧ ↑p * ↑q - (↑p + ↑q) = 231 := by
  sorry

