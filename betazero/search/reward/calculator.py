from betazero.env.ast_parser import get_lean_ast
from betazero.search.graph import ANDORGraph
from betazero.core import Action


class RewardCalculator:
    """Computes environment and dependency rewards, then backs them up through the graph."""

    def __init__(self, W_c: float = 1.0, W_b: float = 0.0, W_m: float = -1.0,
                 W_solve: float = 1.0, gamma: float = 1.0):
        assert W_c > W_b and W_b <= 0 and W_m < W_b, "Required: W_m < W_b <= 0 < W_c"
        self.W_c, self.W_b, self.W_m = W_c, W_b, W_m
        self.W_solve = W_solve
        self.gamma = gamma

    @staticmethod
    def _categorize_nodes(ast_nodes: list, dead_lines: set = None) -> tuple[int, int, int, int]:
        """Classify tactic AST nodes. Returns (total, sorries, junk, dead)."""
        total = sorries = junk = dead = 0
        dead_lines = dead_lines or set()
        for n in ast_nodes:
            kind = n.get("kind", "")
            if not kind or not kind.startswith("Lean.Parser.Tactic."):
                continue
            low = kind.lower()
            if "seq" in low:
                continue
            total += 1
            if "sorry" in low:
                sorries += 1
            elif "skip" in low or "done" in low:
                junk += 1
            elif n.get("pos", {}).get("line") in dead_lines:
                dead += 1
        return total, sorries, junk, dead

    def r_env(self, original_code: str, patched_code: str, verify_result: dict) -> float:
        """Ratio of surviving valid semantic nodes after patching (Section 6.1)."""
        ast_orig = get_lean_ast(original_code)
        tot_orig, sorries_orig, _, _ = self._categorize_nodes(ast_orig)
        if tot_orig == 0:
            return 0.0
        dead_lines = {
            w["pos"]["line"]
            for w in verify_result.get("warnings", [])
            if "unused" in w.get("data", "").lower() or "does nothing" in w.get("data", "").lower()
        }
        ast_patched = get_lean_ast(patched_code)
        tot_patch, sorries_patch, junk_patch, dead_patch = self._categorize_nodes(ast_patched, dead_lines)
        new_sorries = max(0, sorries_patch - sorries_orig)  # sorries added by patcher, not model
        t_valid = max(0, tot_patch - junk_patch - new_sorries - dead_patch)
        return min(1.0, t_valid / tot_orig)

    def r_dep(self, dep_graph: dict) -> float:
        """Weighted dependency reward (Section 6.2)."""
        n_c = len(dep_graph.get("core", []))
        n_b = len(dep_graph.get("benign", []))
        n_m = len(dep_graph.get("malignant", []))
        h_total = n_c + n_b + n_m
        if h_total == 0:
            return 0.0
        return (self.W_c * n_c + self.W_b * n_b + self.W_m * n_m) / h_total

    def compute_returns(self, graph: ANDORGraph) -> dict[Action, float]:
        return graph.backup(gamma=self.gamma, W_solve=self.W_solve)
