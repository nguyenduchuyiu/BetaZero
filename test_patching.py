import json
from gammazero.env.lean_env import LeanEnv
from gammazero.env.lean_verifier import Lean4ServerScheduler
from gammazero.search.sorrifier.sorrifier import Sorrifier

def main():
    candidate_code = """
open BigOperators Real Nat Rat Finset Topology

theorem my_theorem (x y z w : ℕ) 
  (ht : 1 < x ∧ 1 < y ∧ 1 < z) 
  (hw : 0 < w) (h0 : logb ↑x ↑w = 24) 
  (h1 : logb ↑y ↑w = 40) 
  (h2 : logb (↑x * ↑y * ↑z) ↑w = 12) 
  (h_base_x : logb ↑w ↑x = 1 / 24) 
  (h_base_y : logb ↑w ↑y = 1 / 40) 
  : logb (↑w) (↑x * ↑y * ↑z) = 1 / 12 := by
    rw [← inv_inv (logb (↑w) (↑x * ↑y * ↑z))]
    rw [logb_inv]
    rw [h2]
    norm_num
  
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
