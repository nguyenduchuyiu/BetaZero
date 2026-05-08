import json
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from betazero.search.sorrifier.stitcher import ProofStitcher
from betazero.env.expr_parser import get_lean_expr_tree
from betazero.search.sorrifier.dependency_analyzer import SHARED_EXPR_ANALYZER
from betazero.search.reward.calculator import RewardCalculator
from betazero.utils.lean_cmd import build_theorem
from betazero.core.nodes import ProofState

def test_full_reward_flow():
    print("--- 🚀 TESTING FULL REWARD FLOW (STITCH + R_DEP) ---")

    skeleton = """  have ha : a ≠ 0 := h₀.left
  have hb : b ≠ 0 := h₀.right
  have h_ab_nonzero : a * b ≠ 0 := sorry
  have h_frac_sum : a / b + b / a = (a ^ 2 + b ^ 2) / (a * b) := sorry
  have h_target_rewrite : a / b + b / a - a * b = ((a ^ 2 + b ^ 2) / (a * b)) - a * b := sorry
  have h_main : ((a ^ 2 + b ^ 2) / (a * b)) - a * b = 2 := sorry
  have h_final : a / b + b / a - a * b = 2 := sorry
  exact h_final"""

    # Mimic child proofs from tactic agents
    child_proofs = [
        "exact mul_ne_zero ha hb", 
        "field_simp [ha, hb]",
        "rw [h_frac_sum]",
        "by\n  field_simp [h_ab_nonzero]\n  have h2 : (a - b)^2 = (a * b)^2 := by rw [h₁]\n  ring_nf at h2\n  ring_nf\n  linarith [h2, h₁]",
        "rw [h_target_rewrite, h_main]"
    ]

    print("[1] Stitching code...")
    stitched_code = ProofStitcher.stitch(skeleton, child_proofs)
    print("Stitched Code Snippet:\n", stitched_code)

    # Dummy state to build theorem
    state = ProofState(
        context="a b : ℝ\nh₀ : a ≠ 0 ∧ b ≠ 0\nh₁ : a * b = a - b",
        goal="a / b + b / a - a * b = 2",
        header="import Mathlib\nopen Real"
    )
    
    full_code = build_theorem(state, stitched_code)
    # print("Full Compilable Code:\n", full_code)

    print("\n[2] Running Dependency Analysis...")
    results = get_lean_expr_tree(full_code)
    if not results:
        print("❌ Error: Failed to get Expr Tree.")
        return

    root_expr = results[-1].get("expr_value_tree")
    classification = SHARED_EXPR_ANALYZER.classify_skeleton_subgoals(root_expr)
    print("Classification:", json.dumps(classification, indent=2))

    print("\n[3] Calculating r_dep...")
    calculator = RewardCalculator()
    
    mapped_analysis = {
        "core": classification.get("core_solved", []) + classification.get("core_failed", []),
        "benign": classification.get("benign", []),
        "malignant": classification.get("malignant", [])
    }
    
    r_dep_score = calculator.r_dep(mapped_analysis)
    
    # Check for core_failed
    if classification.get("core_failed"):
        print("⚠️ Core subgoals failed! Penalty applied.")
        r_dep_score = -1.0

    print(f"👉 FINAL r_dep SCORE: {r_dep_score}")
    
    # Expecting 1.0 if all are used and solved
    if r_dep_score == 1.0:
        print("✅ SUCCESS: All core subgoals recognized and rewarded!")
    else:
        print("❌ FAILURE: Reward is not 1.0. Check dependencies.")

if __name__ == "__main__":
    test_full_reward_flow()
