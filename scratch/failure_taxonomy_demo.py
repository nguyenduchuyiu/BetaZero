
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from betazero.env.lean_env import Lean4ServerScheduler, LeanEnv
from betazero.search.reward.calculator import RewardCalculator
from betazero.search.sorrifier.sorrifier import Sorrifier
from betazero.utils.lean_cmd import build_theorem
from betazero.core import ProofState

def run_case(name, header, goal, proof_body):
    print(f"\n{'='*20}\n{name}\n{'='*20}")
    state = ProofState(goal=goal, context="", header=header)
    original_code = build_theorem(state, proof_body)
    
    print("--- ORIGINAL CODE (Proof Body) ---")
    print(proof_body)
    
    scheduler = Lean4ServerScheduler(timeout=30)
    lean = LeanEnv(scheduler)
    reward = RewardCalculator()
    sorrifier = Sorrifier(scheduler)
    
    try:
        # 1. Verify original
        before_vr = lean.verify(original_code)
        print("\n--- LEAN DIAGNOSTIC (BEFORE) ---")
        errors = before_vr.get("errors", [])
        for err in errors:
            pos = err.get("pos", {})
            print(f"Error L{pos.get('line')}:{pos.get('column')} - {err.get('data').strip()}")
        
        # 2. Patch
        patched_code = sorrifier.fix_code(original_code)
        from betazero.utils.lean_parse import extract_proof_body
        patched_body = extract_proof_body(patched_code)
        print("\n--- PATCHED CODE (Proof Body) ---")
        print(patched_body)
        
        # 3. Reward
        after_vr = lean.verify(patched_code)
        r_env = reward.r_env(original_code, patched_code, after_vr)
        
        orig_lines = reward._get_clean_proof_lines(original_code)
        patch_lines = reward._get_clean_proof_lines(patched_code)
        
        print("\n--- REWARD BREAKDOWN ---")
        print(f"r_env: {r_env}")
        print(f"Original clean lines: {len(orig_lines)}")
        print(f"Patched clean lines: {len(patch_lines)}")
        print(f"Surviving lines count: {int(r_env * len(orig_lines)) if orig_lines else 0}")
        
    finally:
        scheduler.close()

header = "import Mathlib\n"
goal = "1 + 1 = 2"

if __name__ == "__main__":
    # Case 1: Tactic Parser Error
    run_case("CASE 1: Tactic Parser Error (Unclosed paren)", header, goal, 
             "  have h := 1\n  apply h (")

    # Case 2: Tactic Unresolved Name
    run_case("CASE 2: Tactic Unresolved Name", header, goal, 
             "  apply nonexistent_lemma\n  rfl")

    # Case 3: Tactic Type Mismatch
    run_case("CASE 3: Tactic Type Mismatch", header, goal, 
             "  exact 1")

    # Case 4: Skeleton with correct and incorrect parts
    run_case("CASE 4: Skeleton (Partial Failure)", header, goal, 
             "  have h : 1 + 1 = 2 := by\n    rfl\n  apply nonexistent_lemma\n  exact h")
