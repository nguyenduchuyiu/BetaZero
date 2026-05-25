import difflib
from gammazero.search.graph import ANDORGraph
from gammazero.core import Action
from gammazero.utils.lean_parse import extract_proof_body


class RewardCalculator:
    """Computes environment and dependency rewards, then backs them up through the graph."""

    def __init__(self, W_c: float = 1.0, W_b: float = 0.0, W_m: float = -1.0,
                 W_solve: float = 1.0, gamma: float = 1.0):
        assert W_c > W_b and W_b <= 0 and W_m < W_b, "Required: W_m < W_b <= 0 < W_c"
        self.W_c, self.W_b, self.W_m = W_c, W_b, W_m
        self.W_solve = W_solve
        self.gamma = gamma

    def _get_clean_proof_lines(self, code: str) -> list[str]:
        """Extract the proof body and drop blank/comment lines for survival scoring."""
        proof_body = extract_proof_body(code)
        lines = []
        for line in proof_body.splitlines():
            l = line.strip()
            # Skip blank lines and comments.
            if not l or l.startswith('--') or l.startswith('/-'):
                continue
            lines.append(l)
        return lines

    def r_env(self, original_code: str, patched_code: str, verify_result: dict) -> float:
        """Score quality by the survival rate of original proof lines after patching."""

        orig_lines = self._get_clean_proof_lines(original_code)
        patch_lines = self._get_clean_proof_lines(patched_code)

        if not orig_lines:
            return 0.0

        # 1. Find the longest common subsequence between original and patched.
        # difflib reports exactly which blocks remained untouched.
        matcher = difflib.SequenceMatcher(None, orig_lines, patch_lines)

        # Total original lines preserved verbatim in the patch.
        surviving_lines = sum(match.size for match in matcher.get_matching_blocks())

        # 2. Penalize "dead" lines (verifier warnings: unused / does nothing).
        # Warnings reference patched line numbers; counting them is enough.
        warnings = verify_result.get("warnings", [])
        dead_count = len([w for w in warnings if "unused" in w.get("data", "").lower() or "does nothing" in w.get("data", "").lower()])

        # 3. Base score: only the preserved fraction of the original counts.
        # Numerator capped at len(orig_lines), so the score cannot be inflated.
        # Black-hole replacements are penalized naturally because removed lines
        # do not contribute to surviving_lines.
        valid_survivors = max(0, surviving_lines - dead_count)
        base_score = valid_survivors / len(orig_lines)

        return base_score

    def r_dep(self, dep_graph: dict) -> float:
        """Weighted dependency reward (Section 6.2)."""
        n_c = len(dep_graph.get("core", []))
        n_b = len(dep_graph.get("benign", []))
        n_m = len(dep_graph.get("malignant", []))
        
        if n_c == 0:
            return 0.0
            
        # Apply penalties as denominator weights rather than direct subtractions.
        # - Each benign subgoal adds 0.5 (light penalty: still solvable).
        # - Each malignant subgoal adds 2.0 (heavy penalty: unused and leaves a sorry).
        penalty_b = 0.5
        penalty_m = 2.0
        
        return n_c / (n_c + penalty_b * n_b + penalty_m * n_m)

    def compute_returns(self, graph: ANDORGraph) -> dict[Action, float]:
        return graph.backup(gamma=self.gamma, W_solve=self.W_solve)
