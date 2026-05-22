import json
from gammazero.env.lean_env import LeanEnv
from gammazero.env.lean_verifier import Lean4ServerScheduler
from gammazero.search.sorrifier.sorrifier import Sorrifier

def main():
    candidate_code = """
open BigOperators Real Nat Rat Finset Topology

theorem imo_1977_p5
  (a b q r : ℕ)
  (hp: 0 < a ∧ 0 < b)
  (h₀ : r = (a ^ 2 + b ^ 2) % (a + b))
  (h₁ : q = (a ^ 2 + b ^ 2) / (a + b))
  (h₂ : q ^ 2 + r = 1977) :
  (a = 7 ∧ b = 50) ∨ (a = 37 ∧ b = 50) ∨ (a = 50 ∧ b = 7) ∨ (a = 50 ∧ b = 37) := by
  
    have h_eq : a ^ 2 + b ^ 2 = q * (a + b) + r := by
      rw [h₁, h₀, Nat.div_add_mod]
    have h_r_lt : r < a + b := by
      rw [h₀]; apply Nat.mod_lt; linarith
    have h_q_le : q ≤ 44 := by
      have : q ^ 2 ≤ 1977 := by linarith
      exact Nat.le_of_pow_le_pow_left 2 (by decide) this
    have h_q : q = 44 := by
      by_contra h_ne
      have h_q_max : q ≤ 43 := Nat.le_of_lt_succ (Nat.lt_of_le_of_ne h_q_le h_ne)
      let S : ℤ := (a + b : ℕ)
      have h_r_val : (r : ℤ) = 1977 - (q : ℤ) ^ 2 := by zify at h₂; linarith
      have h_ineq : S ^ 2 ≤ 2 * ((a : ℤ) ^ 2 + (b : 2) ^ 2) := by
        ring_nf; linarith [sq_nonneg (a - b : ℤ)]
      zify [h_eq] at h_ineq
      rw [h_r_val] at h_ineq
      have h_S_bound : (S - q) ^ 2 ≤ 3954 - (q : ℤ) ^ 2 := by ring_nf at h_ineq ⊢; linarith
      have h_r_lt_S : (1977 - (q : ℤ) ^ 2) < S := by zify at h_r_lt; exact h_r_lt
      interval_cases q <;> (zify at h_r_lt_S; nlinarith)
    have h_r : r = 41 := by rw [h_q] at h₂; exact Nat.add_left_cancel h₂
    have h_circle : (a - 22 : ℤ) ^ 2 + (b - 22 : ℤ) ^ 2 = 1009 := by
      zify at h_eq; rw [h_q, h_r] at h_eq; nlinarith
    let X := (a - 22 : ℤ).natAbs
    let Y := (b - 22 : ℤ).natAbs
    have hXY : X ^ 2 + Y ^ 2 = 1009 := by
      zify; rw [Int.sq_natAbs, Int.sq_natAbs, h_circle]
    have h_sol : (X = 15 ∧ Y = 28) ∨ (X = 28 ∧ Y = 15) := by
      have hX_bound : X ≤ 31 := by
        have : X ^ 2 ≤ 1009 := by linarith
        exact Nat.le_of_pow_le_pow_left 2 (by decide) this
      interval_cases X <;> norm_num [hXY] at hXY <;> (try constructor <;> rfl) <;> (try apply Nat.pow_left_inj 2 (by decide); rw [hXY]; norm_num)
    rcases h_sol with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · have ha : (a : ℤ) = 37 ∨ (a : ℤ) = 7 := by
        rcases Int.natAbs_eq_iff.mp (show (a - 22 : ℤ).natAbs = 15 from rfl) with h | h <;> linarith
      have hb : (b : ℤ) = 50 ∨ (b : ℤ) = -6 := by
        rcases Int.natAbs_eq_iff.mp (show (b - 22 : ℤ).natAbs = 28 from rfl) with h | h <;> linarith
      rcases ha with ha | ha <;> rcases hb with hb | hb <;> try { zify at hp; linarith }
      · right; right; right; aesop
      · left; aesop
    · have ha : (a : ℤ) = 50 ∨ (a : ℤ) = -6 := by
        rcases Int.natAbs_eq_iff.mp (show (a - 22 : ℤ).natAbs = 28 from rfl) with h | h <;> linarith
      have hb : (b : ℤ) = 37 ∨ (b : ℤ) = 7 := by
        rcases Int.natAbs_eq_iff.mp (show (b - 22 : ℤ).natAbs = 15 from rfl) with h | h <;> linarith
      rcases ha with ha | ha <;> rcases hb with hb | hb <;> try { zify at hp; linarith }
      · right; right; left; aesop
      · right; left; aesop
  
  """

    scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=60, name="manual_run_exact")
    try:
        lean = LeanEnv(scheduler)
        sorrifier = Sorrifier(scheduler, max_cycles=50)

        print("=== INPUT CODE ===")
        print(candidate_code)
        print("==================\n")

        patched = sorrifier.fix_code(candidate_code)

        print("=== PATCHED CODE ===")
        print(patched)
        print("====================\n")

    finally:
        scheduler.close()

if __name__ == "__main__":
    main()
