import json
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from betazero.env.expr_parser import get_lean_expr_tree
from betazero.search.sorrifier.dependency_analyzer import ExprDependencyAnalyzer

def test_amc_dependency():
    print("--- 🚀 TESTING AMC THEOREM DEPENDENCY ---")

    # The theorem from the user's request
    lean_code = """
import Mathlib
open Real

theorem my_theorem (a b : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : a * b = a - b) : a / b + b / a - a * b = 2 := by
  have ha : a ≠ 0 := h₀.left
  have hb : b ≠ 0 := h₀.right
  have h_ab_nonzero : a * b ≠ 0 := sorry
  have h_frac_sum : a / b + b / a = (a ^ 2 + b ^ 2) / (a * b) := sorry
  have h_target_rewrite : a / b + b / a - a * b = ((a ^ 2 + b ^ 2) / (a * b)) - a * b := sorry
  have h_main : ((a ^ 2 + b ^ 2) / (a * b)) - a * b = 2 := sorry
  have h_final : a / b + b / a - a * b = 2 := sorry
  exact h_final
"""

    results = get_lean_expr_tree(lean_code)
    
    if not results:
        print("❌ Error: No Expr Tree.")
        return

    root_expr = results[-1].get("expr_value_tree")
    analyzer = ExprDependencyAnalyzer()

    print("\n[3] SUBGOAL CLASSIFICATION:")
    classification = analyzer.classify_skeleton_subgoals(root_expr)
    print(json.dumps(classification, indent=2))

if __name__ == "__main__":
    test_amc_dependency()
