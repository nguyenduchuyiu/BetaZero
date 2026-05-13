import json
import sys
from betazero.core.nodes import Action, ProofState
from betazero.env.lean_env import LeanEnv
from betazero.search.reward.calculator import RewardCalculator
from betazero.search.reward.reward_assigner import DependencyRewardAssigner
from betazero.search.sorrifier.stitcher import ProofStitcher
from betazero.utils.lean_cmd import build_theorem

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    data = load_json('outputs/rollouts/gemini3flash/miniF2F-valid-50/aime_1983_p2.json')
    nodes = {n['id']: n for n in data['nodes']}
    
    action_73_data = nodes['action_73']
    state_0_data = nodes['state_0']
    
    # We construct ProofState manually
    # Note: BetaZero's ProofState constructor doesn't take header, we'll just ignore header if it fails
    # Wait, in the JSON, context and goal are in 'content'
    parent_state = ProofState(
        goal=state_0_data['content']['goal'],
        context=state_0_data['content'].get('context', ''),
        # header might be required for build_theorem, let's parse from json if available
    )
    if 'header' in state_0_data['content']:
        parent_state.header = state_0_data['content']['header']
    else:
        # Use a fallback header
        parent_state.header = "open BigOperators Nat Real Topology\n"
        # Wait, the prompt in JSON shows the problem statement
        prompt = action_73_data['prompt']
        # Extract problem from prompt
        import re
        match = re.search(r"\[PROBLEM\]\n```lean4\n(.*?)\n```", prompt, re.DOTALL)
        if match:
            full_prob = match.group(1)
            # The header is everything before 'theorem my_theorem'
            header = full_prob.split('theorem my_theorem')[0]
            parent_state.header = header
            # And we need to use 'theorem my_theorem' signature for build_theorem?
            # Actually build_theorem usually uses parent_state.header and appends theorem signature.
            # But let's look at build_theorem signature in betazero/utils/lean_cmd.py

    action_73 = Action(
        action_type="skeleton",
        extracted_code=action_73_data['extracted_lean_code']
    )
    
    child_ids = ['state_6', 'state_7', 'state_8', 'state_9']
    child_proofs = []
    for cid in child_ids:
        child_proofs.append(nodes[cid]['content'].get('proof_body'))
        
    stitched_code = ProofStitcher.stitch(action_73.extracted_code, child_proofs)
    print("--- STITCHED CODE ---")
    print(stitched_code)
    print("---------------------")
    
    # Let's extract the full theorem code instead of relying on build_theorem if it fails
    # The JSON has the prompt which contains the full problem statement up to `:= by`
    # We can just manually append the stitched code
    match = re.search(r"\[PROBLEM\]\n```lean4\n(.*?:= by\n)  sorry\n```", action_73_data['prompt'], re.DOTALL)
    if match:
        full_code = match.group(1) + stitched_code
    else:
        full_code = build_theorem(parent_state, stitched_code)
    
    print("\n--- FULL CODE ---")
    print(full_code)
    print("-----------------\n")

    print("Initializing LeanEnv & Assigner...")
    # LeanEnv takes a scheduler, but analyze_dependencies doesn't use it, pass None
    lean = LeanEnv(scheduler=None)
    reward_calc = RewardCalculator()
    assigner = DependencyRewardAssigner(lean, reward_calc)
    
    allowed_vars = assigner._extract_sorry_vars(action_73.extracted_code)
    print(f"Extracted allowed sorry vars from skeleton: {allowed_vars}")
    
    print("Analyzing dependencies (calculating r_dep)...")
    dep_analysis = lean.analyze_dependencies(full_code, allowed_vars=allowed_vars)
    
    print("\nDependency Analysis Result:")
    print(json.dumps(dep_analysis, indent=2))
    
    if len(dep_analysis.get("core_failed", [])) > 0:
        r_dep_score = 0.0
    else:
        mapped_analysis = {
            "core": dep_analysis.get("core_solved", []),
            "benign": dep_analysis.get("benign", []),
            "malignant": dep_analysis.get("malignant", [])
        }
        r_dep_score = reward_calc.r_dep(mapped_analysis)
        
    print(f"\n=> FINAL r_dep_score: {r_dep_score}")

if __name__ == "__main__":
    main()
