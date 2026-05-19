import json
from gammazero.env.lean_env import LeanEnv
from gammazero.env.lean_verifier import Lean4ServerScheduler
from gammazero.search.sorrifier.sorrifier import Sorrifier

def main():
    candidate_code = """
open BigOperators Real Nat Rat Finset Topology
theorem my_theorem (f : ℝ → ℝ) 
  (h₀ : ∀ (x : ℝ), f x = x ^ 2 + (18 * x + 30) - 2 * √(x ^ 2 + (18 * x + 45)))
  (h₁ : Fintype ↑(f ⁻¹' {0}))
  (x1 x2 : ℝ)
  (h_equiv : ∀ (x : ℝ), f x = 0 ↔ x ^ 2 + 18 * x + 20 = 0)
  (h_solve : ∀ (x : ℝ), x ^ 2 + 18 * x + 20 = 0 ↔ x = -9 + √61 ∨ x = -9 - √61)
  :{x1, x2} = {-9 + √61, -9 - √61} := by
  have h_subset1 : {x1, x2} ⊆ ({-9 + √61, -9 - √61} : Set ℝ) := sorry
  have h_subset2 : ({-9 + √61, -9 - √61} : Set ℝ) ⊆ {x1, x2} := sorry
  exact Set.Subset.antisymm h_subset1 h_subset2"""

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
