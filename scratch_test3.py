import json
from gammazero.env.lean_env import LeanEnv
from gammazero.env.lean_verifier import Lean4ServerScheduler
from gammazero.search.sorrifier.sorrifier import Sorrifier

def main():
    candidate_code = """open BigOperators Nat Real Topology
theorem my_theorem (x y z w : ℕ) (ht : 1 < x ∧ 1 < y ∧ 1 < z) (hw : 0 < w) (h0 : logb ↑x ↑w = 24) (h1 : logb ↑y ↑w = 40) (h2 : logb (↑x * ↑y * ↑z) ↑w = 12) : logb ↑z ↑w = 60 := by
  have h_recip_x : logb ↑w ↑x = 1 / 24 := by admit
  have h_recip_y : logb ↑w ↑y = 1 / 40 := by
    rw [← h1]
    have hx_gt_1 : 1 < (y : ℝ) := by norm_cast; exact ht.2.1
    have hw_gt_0 : 0 < (w : ℝ) := by norm_cast; exact hw
    have hw_ne_1 : (w : ℝ) ≠ 1 := by
      intro h_w1
      rw [h_w1, logb_one_right] at h1
      norm_num at h1
    rw [logb_base_change _ _ (w : ℝ)]
    · rw [logb_self]
      · field_simp
      · exact hw_gt_0
      · exact hw_ne_1
    · exact hw_gt_0
    · exact hw_ne_1
  have h_recip_xyz : logb ↑w (↑x * ↑y * ↑z) = 1 / 12 := by admit
  have h_log_z_base_w : logb ↑w ↑z = 1 / 60 := by admit
  have h_final_inv : logb ↑z ↑w = 1 / (logb ↑w ↑z) := by admit
  rw [h_final_inv, h_log_z_base_w]
  norm_num"""

    scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=60, name="manual_run_exact")
    try:
        lean = LeanEnv(scheduler)
        sorrifier = Sorrifier(scheduler, max_cycles=10)

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
