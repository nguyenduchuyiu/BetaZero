import json
from gammazero.env.lean_env import LeanEnv
from gammazero.env.lean_verifier import Lean4ServerScheduler
from gammazero.search.sorrifier.sorrifier import Sorrifier

def main():
    candidate_code = """
open BigOperators Real Nat Rat Finset Topology

theorem my_theorem (f : ℕ → ℝ → ℝ) (h₀ : ∀ (x : ℝ), f 1 x = x) (h₁ : ∀ (n : ℕ) (x : ℝ), 0 < n → f (n + 1) x = f n x * (f n x + 1 / ↑n)) : ∃! a, ∀ (n : ℕ), 0 < n → 0 < f n a ∧ f n a < f (n + 1) a ∧ f (n + 1) a < 1 := by
  -- Let C n be the condition on f n a and f (n + 1) a
  let C (a : ℝ) (n : ℕ) : Prop := 0 < f n a ∧ f n a < f (n + 1) a ∧ f (n + 1) a < 1
  
  -- Step 1: Prove monotonicity of f n with respect to a
  have h_mono : ∀ n, 0 < n → StrictMono (f n) := by
    intro n hn
    let P (m : ℕ) : Prop := ∃ (p : Polynomial ℝ), (∀ x, f m x = p.eval x) ∧ (∀ i, 0 ≤ p.coeff i) ∧ 0 < p.coeff 1
    have h_poly : ∀ m, 0 < m → P m := by
      intro m hm
      induction m, hm using Nat.le_induction with
      | base =>
        use Polynomial.X
        simp [h₀, Polynomial.coeff_X]
      | succ m' hm' ih' =>
        obtain ⟨p, hp_ev, hp_c, hp_c1⟩ := ih'
        let p_new := p * p + (Polynomial.C (1 / (m' : ℝ))) * p
        use p_new
        constructor
        · intro x; rw [h₁ m' x hm', hp_ev]; dsimp [p_new]; simp; ring
        · constructor
          · intro i; dsimp [p_new]; rw [Polynomial.coeff_add]
            apply add_nonneg
            · rw [Polynomial.coeff_mul]
              apply Finset.sum_nonneg; intro j hj; apply mul_nonneg (hp_c _) (hp_c _)
            · rw [Polynomial.coeff_C_mul]
              apply mul_nonneg (by simp [hm']) (hp_c i)
          · dsimp [p_new]; rw [Polynomial.coeff_add, Polynomial.coeff_mul, Polynomial.coeff_C_mul]
            apply add_pos_of_nonneg_of_pos
            · apply Finset.sum_nonneg; intro j hj; apply mul_nonneg (hp_c _) (hp_c _)
            · apply mul_pos (by simp [hm']) hp_c1
    obtain ⟨p, hp_ev, hp_c, hp_c1⟩ := h_poly n hn
    intro x y hxy
    rw [hp_ev, hp_ev]
    apply Polynomial.strictMono_of_odd_coeff_and_nonneg_coeff hp_c hp_c1 hxy
  
  -- Step 2: Define I_n as the set of 'a' such that C a k holds for all k ≤ n
  -- We show I_n is a non-empty open interval (L_n, R_n)
  let P (a : ℝ) (n : ℕ) : Prop := ∀ k, 0 < k → k ≤ n → C a k
  have h_interval : ∀ n, 0 < n → ∃ L R, L < R ∧ ∀ a, P a n ↔ L < a ∧ a < R := by admit
  
  -- Step 3: Show that these intervals are nested and their lengths tend to zero
  -- We define L_n and R_n as the sequence of endpoints
  have h_nested_bounds : ∃ (L R : ℕ → ℝ), (∀ n > 0, L n < R n) ∧ 
    (∀ n > 0, ∀ a, P a n ↔ L n < a ∧ a < R n) ∧
    (∀ n > 0, L n < L (n + 1) ∧ R (n + 1) < R n) := by admit
  
  -- Step 4: Show the diameter of the intervals tends to zero to ensure uniqueness
  have h_diameter : ∃ (L R : ℕ → ℝ), (∀ n > 0, ∀ a, P a n ↔ L n < a ∧ a < R n) ∧ 
    Filter.Tendsto (fun n => R n - L n) Filter.atTop (nhds 0) := by admit
  
  -- Step 5: Prove existence and uniqueness using the Nested Interval Theorem logic
  have h_exists_unique : ∃! a, ∀ n, 0 < n → P a n := by admit
  
  -- Final Assembly
  obtain ⟨a, ha_univ, ha_uniq⟩ := h_exists_unique
  use a
  constructor
  · intro n hn
    exact ha_univ n hn n hn (Nat.le_refl n)
  · intro a' ha'
    apply ha_uniq
    intro n hn k hk_pos hk_le
    exact ha' k hk_pos"""

    scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=60, name="manual_run_exact")
    try:
        lean = LeanEnv(scheduler)
        sorrifier = Sorrifier(scheduler, max_cycles=100)

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
