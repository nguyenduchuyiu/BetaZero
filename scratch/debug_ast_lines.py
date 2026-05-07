import os
import json
import sys

# Add the project root to path
sys.path.append("/workspace/npthai/BetaZero")

from betazero.env.ast_parser import get_lean_ast
from betazero.utils.lean_cmd import build_theorem
from betazero.core import ProofState

# Mimic the state in action_175
state = ProofState(
    context="a b : ℝ\nh₀ : logb 8 a + logb 4 (b ^ 2) = 5\nh₁ : logb 8 b + logb 4 (a ^ 2) = 7\nh_log8_4 : logb 8 4 = 2 / 3\nh_eq1' : logb 8 (b ^ 2) = 2 * logb 8 b / logb 8 4\nh_eq2' : logb 8 (a ^ 2) = 2 * logb 8 a / logb 8 4\nh_rewrite0 : logb 8 a + 2 * logb 8 b / logb 8 4 = 5\nh_rewrite1 : logb 8 b + 2 * logb 8 a / logb 8 4 = 7\nh_subst0 : logb 8 a + 3 * logb 8 b = 5\nh_subst1 : logb 8 b + 3 * logb 8 a = 7\nh_log_a : logb 8 a = 2",
    goal="logb 8 b = 1",
    header="open BigOperators Nat Real Topology"
)
code = "\n  linarith"

full_code = build_theorem(state, code)
print(f"--- Full Code ---\n{full_code}\n-----------------")

ast = get_lean_ast(full_code)
for node in ast:
    kind = node.get("kind", "")
    if kind.startswith("Lean.Parser.Tactic."):
        print(f"Tactic: {kind}, Line: {node.get('pos', {}).get('line')}")
