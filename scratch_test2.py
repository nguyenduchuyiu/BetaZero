import json
import re
from pathlib import Path
from gammazero.core import ProofState, Action
from gammazero.env.lean_env import LeanEnv
from gammazero.env.lean_verifier import Lean4ServerScheduler
from gammazero.policy.output_parser import get_subgoal_tactic_code
from gammazero.search.rollout.batch_executor import BatchExecutor
from gammazero.search.reward.calculator import RewardCalculator
from gammazero.search.sorrifier.sorrifier import Sorrifier
from gammazero.utils.lean_cmd import build_theorem

def main():
    # 1. Define parent state
    context = (
        "x y z w : ℕ\n"
        "ht : 1 < x ∧ 1 < y ∧ 1 < z\n"
        "hw : 0 < w\n"
        "h0 : logb ↑x ↑w = 24\n"
        "h1 : logb ↑y ↑w = 40\n"
        "h2 : logb (↑x * ↑y * ↑z) ↑w = 12"
    )
    goal = "logb ↑z ↑w = 60"
    header = "open BigOperators Nat Real Topology"
    parent_state = ProofState(context, goal, header)

    # 2. Define skeleton
    skeleton_body = (
        "  have h_recip_x : logb ↑w ↑x = 1 / 24 := sorry\n"
        "  have h_recip_y : logb ↑w ↑y = 1 / 40 := sorry\n"
        "  have h_recip_xyz : logb ↑w (↑x * ↑y * ↑z) = 1 / 12 := sorry\n"
        "  have h_log_z_base_w : logb ↑w ↑z = 1 / 60 := sorry\n"
        "  have h_final_inv : logb ↑z ↑w = 1 / (logb ↑w ↑z) := sorry\n"
        "  rw [h_final_inv, h_log_z_base_w]\n"
        "  norm_num"
    )
    dummy = ProofState("''", 'dummy', header)
    skeleton = Action('skeleton', "''", skeleton_body, children=(dummy,)*5)

    # 3. Define original tactic code (failing)
    orig = """rw [← h1]
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
· exact hw_ne_1"""

    # 4. Initialize Lean environment & Sorrifier
    scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=60, name="manual_patch_run")
    try:
        lean = LeanEnv(scheduler)
        sorrifier = Sorrifier(scheduler, max_cycles=50)

        # 5. Build full candidate code and run sorrifier
        skeleton_code = BatchExecutor._subgoal_skeleton_with_replacement(skeleton, 1, orig)
        full_candidate = build_theorem(parent_state, skeleton_code)
        
        print("=== BEFORE PATCHING (CANDIDATE CODE) ===")
        print(full_candidate)
        print("========================================\n")
        
        patched = sorrifier.fix_code(full_candidate)
        
        print("=== AFTER PATCHING (FULL PATCHED THEOREM) ===")
        print(patched)
        print("=============================================\n")

        # 6. Extract patched action code (only the subgoal part)
        patched_action_code = get_subgoal_tactic_code(
            f"```lean4\n{patched}\n```",
            skeleton.extracted_code,
            1
        )
        print("=== EXTRACTED PATCHED ACTION CODE ===")
        print(patched_action_code)
        print("=====================================\n")

        # 7. Compute target score code
        fo = BatchExecutor._subgoal_target_score_code(parent_state, skeleton, 1, orig)
        fp = BatchExecutor._subgoal_target_score_code(parent_state, skeleton, 1, patched_action_code)

        print("=== ORIGINAL SCORE TARGET ===")
        print(fo)
        print("=============================\n")

        print("=== PATCHED SCORE TARGET ===")
        print(fp)
        print("============================\n")

        # 8. Calculate r_env
        reward_calculator = RewardCalculator()
        patched_vr = lean.verify(patched)
        r_env = reward_calculator.r_env(fo, fp, patched_vr)
        print(f"r_env Score: {r_env}")

    finally:
        scheduler.close()

if __name__ == "__main__":
    main()
