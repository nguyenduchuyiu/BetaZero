"""
RewardCalculator.

r_env  : Syntactic reward (valid semantic AST nodes ratio).
r_dep  : Structural reward (weighted core/benign/malignant).
compute_returns : Backups ANDORGraph to produce Q-values.
"""

from ast_parser import get_lean_ast
from lean_verifier import verify_lean_code
from and_or_graph import ANDORGraph
from nodes import Action

class RewardCalculator:
    def __init__(self, W_c: float = 1.0, W_b: float = 0.0, W_m: float = -1.0,
                 W_solve: float = 1.0, gamma: float = 1.0):
        assert W_c > W_b, "W_c must be > W_b"
        assert W_b <= 0, "W_b must be <= 0"
        assert W_m < W_b, "W_m must be < W_b"
        self.W_c = W_c
        self.W_b = W_b
        self.W_m = W_m
        self.W_solve = W_solve
        self.gamma = gamma

    @staticmethod
    def _categorize_nodes(ast_nodes: list, dead_lines: set = None) -> tuple[int, int, int, int]:
        """
        Phân loại AST node.
        Trả về: (total_tactics, sorries, junk, dead)
        """
        total = sorries = junk = dead = 0
        dead_lines = dead_lines or set()

        for n in ast_nodes:
            kind = n.get("kind", "")
            if not kind or not kind.startswith("Lean.Parser.Tactic."):
                continue
                
            low = kind.lower()
            # Bỏ qua node cấu trúc
            if "seq" in low:
                continue
                
            total += 1 # Tính mọi nỗ lực gõ lệnh vào tổng
            
            if "sorry" in low:
                sorries += 1
            elif "skip" in low or "done" in low:
                junk += 1  # Lệnh rác chủ động
            elif n.get("pos", {}).get("line") in dead_lines:
                dead += 1  # Lệnh rác bị linter tóm
                
        return total, sorries, junk, dead

    def r_env(self, original_code: str, patched_code: str, verify_result: dict) -> float:
        # 1. Đánh giá nỗ lực ban đầu của Model
        ast_orig = get_lean_ast(original_code)
        print(f"ast_orig: {ast_orig}")
        tot_orig, sorries_orig, _, _ = self._categorize_nodes(ast_orig)
        print(f"tot_orig: {tot_orig}")
        print(f"sorries_orig: {sorries_orig}")
        if tot_orig == 0:
            return 0.0

        # 2. Quét Linter để tìm dead code
        dead_lines = {
            w["pos"]["line"] 
            for w in verify_result.get("warnings", []) 
            if "unused" in w.get("data", "").lower() or "does nothing" in w.get("data", "").lower()
        }

        # 3. Đánh giá chất lượng code sống sót sau khi vá
        ast_patched = get_lean_ast(patched_code)
        tot_patch, sorries_patch, junk_patch, dead_patch = self._categorize_nodes(ast_patched, dead_lines)
        print(f"ast_patched: {ast_patched}")
        print(f"tot_patch: {tot_patch}")
        print(f"sorries_patch: {sorries_patch}")
        print(f"junk_patch: {junk_patch}")
        print(f"dead_patch: {dead_patch}")
        # Số lượng sorry do Sorrifier ĐẺ THÊM ra để vá lỗi (không tính sorry model tự viết)
        new_sorries = max(0, sorries_patch - sorries_orig)

        # 4. Công thức chốt: Lấy tổng sống sót trừ đi Rác, Lỗi, và Deadcode
        print(f"new_sorries: {new_sorries}")
        t_valid = tot_patch - junk_patch - new_sorries - dead_patch
        print(f"t_valid: {t_valid}")
        t_valid = max(0, t_valid)

        return min(1.0, t_valid / tot_orig)

    def r_dep(self, dep_graph: dict) -> float:
        n_c = len(dep_graph.get("core", []))
        n_b = len(dep_graph.get("benign", []))
        n_m = len(dep_graph.get("malignant", []))
        h_total = n_c + n_b + n_m
        
        if h_total == 0:
            return 0.0
            
        return (self.W_c * n_c + self.W_b * n_b + self.W_m * n_m) / h_total

    def compute_returns(self, graph: ANDORGraph) -> dict[Action, float]:
        return graph.backup(gamma=self.gamma, W_solve=self.W_solve)