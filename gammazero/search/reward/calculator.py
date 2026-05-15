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
        """Extract proof body, then drop blank/comment lines for survival scoring."""
        proof_body = extract_proof_body(code)
        lines = []
        for line in proof_body.splitlines():
            l = line.strip()
            # Bỏ qua dòng trống và comment
            if not l or l.startswith('--') or l.startswith('/-'):
                continue
            lines.append(l)
        return lines

    def r_env(self, original_code: str, patched_code: str, verify_result: dict) -> float:
        """Đánh giá chất lượng dựa trên Tỷ lệ Code Gốc Sống Sót (Survival Rate)."""
        
        orig_lines = self._get_clean_proof_lines(original_code)
        patch_lines = self._get_clean_proof_lines(patched_code)
        
        if not orig_lines:
            return 0.0

        # 1. Tìm Longest Common Subsequence (LCS) giữa code gốc và code vá
        # difflib sẽ tìm chính xác những block code nào không bị đụng tới.
        matcher = difflib.SequenceMatcher(None, orig_lines, patch_lines)
        
        # Tổng số dòng gốc CÒN NGUYÊN VẸN trong bản vá
        surviving_lines = sum(match.size for match in matcher.get_matching_blocks())

        # 2. Xử lý "Dead lines" (Cảnh báo unused/does nothing từ verifier)
        # Lưu ý: Cảnh báo thường trỏ vào line number của bản vá, 
        # ta cần đếm số lượng cảnh báo để phạt trừ lùi.
        warnings = verify_result.get("warnings", [])
        dead_count = len([w for w in warnings if "unused" in w.get("data", "").lower() or "does nothing" in w.get("data", "").lower()])

        # 3. TÍNH BASE SCORE: Điểm chỉ đến từ phần code gốc được giữ lại
        # Tử số bị giới hạn tối đa = len(orig_lines). Không thể lạm phát.
        # Black Hole bị phạt tự nhiên vì phần bị xóa sẽ không được tính vào surviving_lines.
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
            
        # Thay vì trừ điểm trực tiếp, ta dùng hệ số phạt ở mẫu số.
        # - Mỗi benign subgoal làm tăng mẫu số thêm 0.5 (phạt nhẹ vì dù sao cũng giải được)
        # - Mỗi malignant subgoal làm tăng mẫu số thêm 2.0 (phạt nặng vì rác và để lại sorry)
        penalty_b = 0.5
        penalty_m = 2.0
        
        return n_c / (n_c + penalty_b * n_b + penalty_m * n_m)

    def compute_returns(self, graph: ANDORGraph) -> dict[Action, float]:
        return graph.backup(gamma=self.gamma, W_solve=self.W_solve)
