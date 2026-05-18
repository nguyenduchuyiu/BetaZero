import json
from gammazero.env.lean_env import LeanEnv
from gammazero.env.lean_verifier import Lean4ServerScheduler
from gammazero.search.reward.reward_assigner import DependencyRewardAssigner
from gammazero.search.reward.calculator import RewardCalculator
from gammazero.core.nodes import ProofState, Action
from gammazero.utils.lean_cmd import build_theorem

def main():
    # 1. Define parent state (using root state_0)
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

    # 2. Define target patched tactic code with "safe" local sorry (inside have)
    # hw_gt_0 is fully proved, hw_ne_1 is sorrified, but the rest of the proof works!
    patched_action_code = """rw [← h1]
have hx_gt_1 : 1 < (y : ℝ) := by norm_cast; exact ht.2.1
have hw_gt_0 : 0 < (w : ℝ) := by norm_cast; exact hw
have hw_ne_1 : (w : ℝ) ≠ 1 := by sorry
rw [logb_base_change _ _ (w : ℝ)]
· rw [logb_self]
  · field_simp
  · exact hw_gt_0
  · exact hw_ne_1
· exact hw_gt_0
· exact hw_ne_1"""

    # 3. Assemble full theorem
    # In this case, the sibling subgoals are filled with `by admit`
    # so the theorem itself compiles successfully.
    full_tactic_body = f"""  have h_recip_x : logb ↑w ↑x = 1 / 24 := by admit
  have h_recip_y : logb ↑w ↑y = 1 / 40 := by
{chr(10).join("    " + l for l in patched_action_code.splitlines())}
  have h_recip_xyz : logb ↑w (↑x * ↑y * ↑z) = 1 / 12 := by admit
  have h_log_z_base_w : logb ↑w ↑z = 1 / 60 := by admit
  have h_final_inv : logb ↑z ↑w = 1 / (logb ↑w ↑z) := by admit
  rw [h_final_inv, h_log_z_base_w]
  norm_num"""

    full_code = build_theorem(parent_state, full_tactic_body)

    scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=60, name="manual_rdep_run")
    try:
        lean = LeanEnv(scheduler)
        reward_calculator = RewardCalculator()
        assigner = DependencyRewardAssigner(lean, reward_calculator)

        print("=== FULL GENERATED CODE ===")
        print(full_code)
        print("============================\n")

        print("=== VERIFYING FULL CODE ===")
        vr = lean.verify(full_code)
        print("Verification result 'complete':", vr.get("complete"))
        print("Verification result 'pass':", vr.get("pass"))
        print("Verification errors:")
        print(json.dumps(vr.get("errors", []), indent=2))

        print("\n=== ANALYZING DEPENDENCIES ===")
        local_vars = assigner._extract_action_local_vars(patched_action_code)
        print("Extracted Local Variables:", local_vars)
        
        dep_analysis = lean.analyze_dependencies(
            full_code,
            allowed_vars=local_vars,
            target_name="my_theorem.h_recip_y"
        )
        print("Dependency Analysis Result:")
        print(json.dumps(dep_analysis, indent=2))

        # Calculate r_dep
        r_dep = assigner.calculate_patched_tactic_r_dep(
            full_code,
            patched_action_code,
            target_name="my_theorem.h_recip_y"
        )
        print(f"\nFinal calculate_patched_tactic_r_dep: {r_dep}")

    finally:
        scheduler.close()

if __name__ == "__main__":
    main()
