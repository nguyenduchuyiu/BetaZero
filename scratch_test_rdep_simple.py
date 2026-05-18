import json
from gammazero.env.lean_env import LeanEnv
from gammazero.env.lean_verifier import Lean4ServerScheduler
from gammazero.search.reward.reward_assigner import DependencyRewardAssigner
from gammazero.search.reward.calculator import RewardCalculator
from gammazero.core.nodes import ProofState
from gammazero.utils.lean_cmd import build_theorem

def main():
    # 1. Define a simple parent state
    context = (
        "A B : Prop\n"
        "hA : A\n"
        "hB : B"
    )
    goal = "A ∧ B"
    header = "open BigOperators Nat Real Topology"
    parent_state = ProofState(context, goal, header)

    # 2. Define tactic code with a safe sorry (inside hA_copy) and local variables
    patched_action_code = """have hA_copy : A := by sorry
have hB_copy : B := by exact hB
exact ⟨hA_copy, hB_copy⟩"""

    full_code = build_theorem(parent_state, patched_action_code)

    scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=60, name="manual_rdep_simple")
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
            target_name=None
        )
        print("Dependency Analysis Result:")
        print(json.dumps(dep_analysis, indent=2))

        # Calculate r_dep
        r_dep = assigner.calculate_patched_tactic_r_dep(
            full_code,
            patched_action_code,
            target_name=None
        )
        print(f"\nFinal calculate_patched_tactic_r_dep: {r_dep}")

    finally:
        scheduler.close()

if __name__ == "__main__":
    main()
