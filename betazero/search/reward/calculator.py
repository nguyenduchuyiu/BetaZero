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
        total = sorries = junk = dead = 0
        dead_lines = dead_lines or set()
        
        # Tập hợp các từ khóa nhận diện node cấu trúc, không phải tactic thực thi
        STRUCTURAL_KEYWORDS = {
            "seq",          # tacticSeq, tacticSeq1Indented, tacticSeqBracketed
            "bytactic",     # Chữ 'by'
            "focus",        # Dấu chấm focus '·'
            "evaltactic",   # Wrapper
            "tacticblock",  # Wrapper
            "paren"         # Ngoặc
        }

        for n in ast_nodes:
            kind = n.get("kind", "")
            
            # Chỉ bắt các node có mùi tactic
            if not kind or "tactic" not in kind.lower():
                continue
                
            low = kind.lower()
            
            # CHẶN TẤT CẢ CÁC NODE CẤU TRÚC
            if any(k in low for k in STRUCTURAL_KEYWORDS):
                continue
                
            total += 1
            if "sorry" in low:
                sorries += 1
            elif "skip" in low or "done" in low:
                junk += 1
            elif n.get("pos", {}).get("line") in dead_lines:
                dead += 1
                
        return total, sorries, junk, dead

    def _count_intended_steps(self, code: str) -> int:
        """Đếm số lượng lệnh 'have/let/exact/calc' thực tế có trong văn bản."""
        # Chỉ đếm các dòng bắt đầu bằng từ khóa tactic chính
        lines = [l.strip() for l in code.splitlines() if l.strip()]
        count = 0
        keywords = ['have', 'let', 'exact', 'calc', 'set', 'by', 'intro']
        for l in lines:
            # Check xem dòng đó có bắt đầu bằng keyword không (bỏ qua space/comment)
            if any(l.startswith(k + " ") or l == k for k in keywords):
                count += 1
        return count

    def r_env(self, original_code: str, patched_code: str, verify_result: dict) -> float:
        """Tỷ lệ các node ngữ nghĩa còn sống sót sau khi vá lỗi."""
        # 1. Lấy AST gốc
        ast_orig = get_lean_ast(original_code)
        tot_orig_ast, sorries_orig, _, _ = self._categorize_nodes(ast_orig)
        
        # 2. CHỐT CHẶN: Dùng Regex để lấy số lượng lệnh "dự kiến"
        # Nếu AST bị sập và trả về ít node hơn thực tế, ta lấy con số lớn hơn
        intended_steps = self._count_intended_steps(original_code)
        tot_orig = max(tot_orig_ast, intended_steps)
        
        if tot_orig == 0:
            return 0.0
        
        # 3. Phân tích bản vá (Bản vá thường đã được Sorrifier làm cho 'hợp lệ' về AST)
        dead_lines = set()
        for w in verify_result.get("warnings", []):
            msg = w.get("data", "").lower()
            if "unused" in msg or "does nothing" in msg:
                dead_lines.add(w["pos"]["line"])
                
        ast_patched = get_lean_ast(patched_code)
        tot_patch, sorries_patch, junk_patch, dead_patch = self._categorize_nodes(ast_patched, dead_lines)
        
        # Tính số sorry mới sinh ra do máy vá (không phải do model chủ động viết)
        new_sorries = max(0, sorries_patch - sorries_orig)
        
        # Lệnh hợp lệ = Tổng node trong bản vá - (rác + sorry mới + code thừa)
        t_valid = max(0, tot_patch - junk_patch - new_sorries - dead_patch)
        
        # 4. Trả về tỷ lệ dựa trên mẫu số "thực"
        return min(1.0, t_valid / tot_orig)

    def r_dep(self, dep_graph: dict) -> float:
        """Weighted dependency reward (Section 6.2)."""
        n_c = len(dep_graph.get("core", []))
        n_b = len(dep_graph.get("benign", []))
        n_m = len(dep_graph.get("malignant", []))
        
        if n_c == 0:
            return 0.0
            
        # Thay vì trừ điểm trực tiếp, ta dùng hệ số phạt ở mẫu số.
        # - Mỗi benign subgoal làm tăng mẫu số thêm 0.5 (phạt nhẹ vì dù sao cũng giải được)
        # - Mỗi malignant subgoal làm tăng mẫu số thêm 2.0 (phạt nặng vì rác và để lại sorry)
        penalty_b = 0.5
        penalty_m = 2.0
        
        return n_c / (n_c + penalty_b * n_b + penalty_m * n_m)

    def compute_returns(self, graph: ANDORGraph) -> dict[Action, float]:
        return graph.backup(gamma=self.gamma, W_solve=self.W_solve)
