"""
AST-Based Automated Proof Patcher for Lean 4
--------------------------------------------
This module automates the process of fixing broken Lean 4 proofs by replacing 
faulty tactics with the `sorry` axiom. 

Architecture:
1. AST-Guided Truncation: Uses Lean's AST to precisely locate tactic boundaries.
2. Indentation Heuristics: Infers structural hierarchy where AST lacks context (e.g., closing scopes).
3. Oscillation Fallback: Detects infinite correction loops caused by Lean's syntax 
   intolerance and resets the parent block to prevent halting.
"""

import sys
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
from lean_verifier import Lean4ServerScheduler
from ast_parser import get_lean_ast

BLOCK_STARTERS = (
    "have", "·", ".", "cases ", "cases' ", "induction ", 
    "induction' ", "rintro ", "intro ", "calc", "match", 
    "lemma", "theorem", "def"
)

TRIVIAL_TACTICS = frozenset({"skip", "done", "trivial", "decide", "rfl"})


class Sorrifier:
    def __init__(self, code: str,
                 repl_verifier: Lean4ServerScheduler,
                 max_cycles: int = 50, 
                 log_path: Optional[str] = None,
                 verify_timeout: int = 60):
        self.current_content = self._strip_noop_tactics(code)
        self.max_cycles = max_cycles
        self.log_path = log_path
        self._last_action_msg = ""
        self.repl_verifier = repl_verifier
        self.verify_timeout = verify_timeout

    def fix_code(self) -> str:
        """Iteratively patch Lean 4 errors until the code compiles or max_cycles is reached."""
        tqdm.write("Starting Auto Sorrifier")
        seen_states = set()
        self._log_state("Initial State")

        with tqdm(total=self.max_cycles, desc="Processing", unit="cycle") as pbar:
            for _ in range(self.max_cycles):
                try:
                    fatal_errors, unsolved_goals = self._get_lean_errors()
                except RuntimeError as e:
                    tqdm.write(f"\nHALTED: {e}")
                    self.current_content = self._force_full_sorrify()
                    tqdm.write("Timeout/crash fallback: force full sorrify.")
                    return self.current_content

                if not fatal_errors and not unsolved_goals:
                    tqdm.write("\nSUCCESS: File is fully compiled with sorries.")
                    return self.current_content

                is_fatal = bool(fatal_errors)
                err_line, err_msg = fatal_errors[0] if is_fatal else unsolved_goals[0]
                if not self._is_valid_line_number(err_line):
                    try:
                        fatal_errors, unsolved_goals = self._get_lean_errors()
                        is_fatal = bool(fatal_errors)
                        err_line, err_msg = fatal_errors[0] if is_fatal else unsolved_goals[0]
                    except RuntimeError:
                        pass
                err_line = self._normalize_line_number(err_line)

                if self.current_content in seen_states:
                    tqdm.write(f"\nOscillation detected at line {err_line}. Triggering Parent Block Reset...")
                    try:
                        self._resolve_infinite_loop(err_line)
                    except IndexError as e:
                        tqdm.write(f"Index error during oscillation fallback: {e}. Force full sorrify.")
                        self.current_content = self._force_full_sorrify()
                        return self.current_content
                    self._log_state("Fallback: Oscillation Resolution")
                    pbar.update(1)
                    continue

                seen_states.add(self.current_content)
                pbar.set_postfix_str(f"{'Fatal' if is_fatal else 'Unsolved'} @ L{err_line}")

                try:
                    success = self._apply_normal_fix(err_line, is_fatal, err_msg)
                except IndexError as e:
                    tqdm.write(f"Index error during normal fix: {e}. Force full sorrify.")
                    self.current_content = self._force_full_sorrify()
                    return self.current_content
                if not success:
                    tqdm.write(f"\nHALTED: Unrecoverable error at line {err_line}.")
                    break

                self._log_state(self._last_action_msg)
                pbar.update(1)

        return self.current_content

    # ==========================================
    # CORE FIXING LOGIC
    # ==========================================

    def _resolve_infinite_loop(self, err_line: int):
        """
        Fallback resolution for correction oscillations.
        """
        lines = self.current_content.splitlines()
        err_line = self._normalize_line_number(err_line, total_lines=len(lines))
        original_content = self.current_content # Lưu lại trạng thái cũ để so sánh
        
        # 1. Search backward for nearest parent block by string match
        boss_idx = -1
        for i in range(err_line - 1, -1, -1):
            line_str = lines[i].strip()
            if any(line_str.startswith(kw) for kw in ["have ", "lemma ", "theorem ", "def ", "·", "cases ", "match "]):
                boss_idx = i
                break
        
        if boss_idx != -1:
            boss_line = lines[boss_idx]
            boss_indent = len(boss_line) - len(boss_line.lstrip())
            
            # 2. Replace parent block body with sorry, retain declaration
            if ":=" in boss_line:
                lines[boss_idx] = boss_line.split(":=")[0] + ":= by sorry"
            elif boss_line.strip().startswith("·"):
                lines[boss_idx] = " " * boss_indent + "· sorry"
            elif "=>" in boss_line:
                lines[boss_idx] = boss_line.split("=>")[0] + "=> sorry"
            
            tqdm.write(f"Reset parent block at line {boss_idx + 1}")
            
            # 3. Remove all child lines (greater indent) following parent
            i = boss_idx + 1
            while i < len(lines):
                if not lines[i].strip():
                    lines[i] = ""
                    i += 1
                    continue
                curr_indent = len(lines[i]) - len(lines[i].lstrip())
                if curr_indent > boss_indent:
                    lines[i] = ""
                    i += 1
                else:
                    break
        else:
            tqdm.write("Parent block not found, deleting problematic line.")
            if err_line - 1 < len(lines):
                lines[err_line - 1] = ""
            
        self.current_content = self._clean_redundant_sorries(lines)
        
        # 4. Deadlock Breaker: 
        # Nếu logic phía trên không làm code thay đổi (ví dụ parent đã bị sorry từ trước 
        # và child không bị xóa do khác indent), ta ép buộc xóa bỏ dòng gây lỗi.
        if self.current_content == original_content:
            tqdm.write(f"Fallback didn't mutate code! Force deleting error line {err_line}.")
            if err_line - 1 < len(lines):
                lines[err_line - 1] = ""
            self.current_content = self._clean_redundant_sorries(lines)

    def _apply_normal_fix(self, error_line: int, is_fatal: bool, err_msg: str) -> bool:
        lines = self.current_content.splitlines()
        error_line = self._normalize_line_number(error_line, total_lines=len(lines))

        # 1. Xử lý Trivial Tactics (Spam rác)
        line_content = lines[error_line - 1].strip()
        if line_content in TRIVIAL_TACTICS:
            lines[error_line - 1] = ""
            self._last_action_msg = f"Removed failing trivial tactic '{line_content}' at L{error_line}"
            tqdm.write(self._last_action_msg)
            self.current_content = self._clean_redundant_sorries(lines)
            return True

        blocks = self._get_ast_lines()
        enclosing = [b for b in blocks if b["start_line"] <= error_line <= b["end_line"]]

        def emergency_fallback():
            msg = f"AST parsing failed at L{error_line}. Applying basic single-line replacement."
            tqdm.write(msg)
            self._last_action_msg = msg
            indent = len(lines[error_line - 1]) - len(lines[error_line - 1].lstrip())
            lines[error_line - 1] = " " * indent + "sorry"
            self.current_content = "\n".join(lines) + "\n"
            return True

        # 2. Xử lý Lỗi Cú pháp / Logic sai (Fatal Error)
        if is_fatal:
            valid_nodes = [b for b in enclosing if "tactic" in b["kind"].lower() or "seq" in b["kind"].lower()]
            if not valid_nodes: return emergency_fallback()
            
            target = min(valid_nodes, key=lambda x: x["end_line"] - x["start_line"])
            L_start, L_end = target["start_line"], target["end_line"]
            start_line_str = lines[L_start - 1]
            
            is_orphan_error = "no goals" in err_msg.lower() or "goals accomplished" in err_msg.lower()
            
            # --- Tách mảng an toàn ---
            new_lines = lines[:L_start - 1]
            indent = len(start_line_str) - len(start_line_str.lstrip())
            
            if is_orphan_error:
                # Lỗi Orphan: Tactic bị thừa vì goal đã đóng. 
                # -> XÓA SẠCH, KHÔNG CHÈN SORRY.
                self._last_action_msg = f"Removed orphaned tactic [{target['kind']}] L{L_start}..L{L_end}"
                # Bỏ qua đoạn code thừa, nối thẳng phần đuôi vào
                new_lines.extend(lines[L_end:])
                
            elif self._is_block_starter(start_line_str) and ":=" in start_line_str:
                # Lỗi thủng Block (have/let): Truncate toàn bộ body của nó
                self._last_action_msg = f"Hollowed out block [{target['kind']}] starting at L{L_start}"
                clean_header = start_line_str.split(":=")[0] + ":= by sorry"
                new_lines.append(clean_header)
                new_lines.extend(lines[L_end:])
                
            else:
                # Lỗi tactic lá thông thường: Thay node đó bằng sorry
                self._last_action_msg = f"Replaced leaf tactic [{target['kind']}] L{L_start}..L{L_end}"
                new_lines.append(" " * indent + "sorry")
                new_lines.extend(lines[L_end:])
                
            tqdm.write(self._last_action_msg)
            self.current_content = "\n".join(new_lines) + "\n"
                
        # 3. Xử lý Chưa chứng minh xong (Unsolved Goals)
        else: 
            scopes = ["declaration", "tactichave", "tacticcases", "tacticmatch", "tacticlet"]
            valid_nodes = [b for b in enclosing if any(s in b["kind"].lower() for s in scopes)]
            
            if not valid_nodes:
                valid_nodes = [b for b in enclosing if "seq" in b["kind"].lower() or "bytactic" in b["kind"].lower()]
                if not valid_nodes: return emergency_fallback()
                target = max(valid_nodes, key=lambda x: x["end_line"] - x["start_line"])
            else:
                target = min(valid_nodes, key=lambda x: x["end_line"] - x["start_line"])

            L_start, L_end = target["start_line"], target["end_line"]
            
            # --- FIX LOGIC INDENT ---
            # Default fallback: thụt vào 2 space so với dòng cha (block starter)
            parent_indent = len(lines[L_start - 1]) - len(lines[L_start - 1].lstrip())
            indent = parent_indent + 2 
            
            # Cố gắng dò tìm indent của thằng con đầu tiên (nếu có)
            for i in range(L_start, L_end): # Bỏ qua dòng cha (L_start-1)
                line = lines[i]
                if line.strip() and not line.strip().startswith("--"):
                    indent = len(line) - len(line.lstrip())
                    break

            self._last_action_msg = f"Closed scope [{target['kind']}] at L{L_end} (Indent: {indent})"
            tqdm.write(self._last_action_msg)
            
            # Chèn sorry vào cuối scope (không xóa code cũ)
            new_lines = lines[:L_end]
            new_lines.append(" " * indent + "sorry")
            new_lines.extend(lines[L_end:])
            
            self.current_content = "\n".join(new_lines) + "\n"

        self.current_content = self._clean_redundant_sorries(self.current_content.splitlines())
        return True

    def _get_lean_errors(self) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
        """
        Dùng Lean4ServerScheduler (`repl_verifier`) để chạy `verify_lean_code`
        trong background process, sau đó phân loại lỗi.
        """
        req_ids = self.repl_verifier.submit_all_request(
            [dict(code=self.current_content, timeout=self.verify_timeout)]
        )
        result = self.repl_verifier.get_all_request_outputs(req_ids)[0]
        print(f"[REPL] verify_lean_code executed in {result.get('verify_time', 0):.4f} seconds")

        if result.get("system_errors"):
            raise RuntimeError(f"Lean verification timed out or crashed: {result['system_errors'][:200]}")

        fatal_errors: List[Tuple[int, str]] = []
        unsolved_goals: List[Tuple[int, str]] = []

        for msg in result.get("errors", []):
            ln = msg.get("pos", {}).get("line", 1)
            txt = msg.get("data", "")
            if "unsolved goals" in txt:
                unsolved_goals.append((ln, txt))
            else:
                fatal_errors.append((ln, txt))

        return sorted(fatal_errors), sorted(unsolved_goals)

    def _get_ast_lines(self) -> List[Dict]:
        """Fetch AST blocks and convert byte offsets to line numbers."""
        blocks = get_lean_ast(self.current_content)

        raw_bytes = self.current_content.encode('utf-8')
        for b in blocks:
            b["start_line"] = self._byte_to_line(raw_bytes, b["start_byte"])
            b["end_line"] = self._byte_to_line(raw_bytes, b["end_byte"])

        return blocks

    def _clean_redundant_sorries(self, lines: List[str]) -> str:
        """
        Removes duplicated `sorry` lines and empty lines generated during automated fixes.
        """
        cleaned = []
        for line in lines:
            if line == "":
                continue
            stripped = line.strip()
            if stripped == "sorry" and cleaned and cleaned[-1].strip() == "sorry":
                continue
            cleaned.append(line)
            
        return "\n".join(cleaned) + "\n"

    def _force_full_sorrify(self) -> str:
        """
        On verifier timeout/crash, collapse the proof body to a single `sorry`.
        """
        marker = ":= by"
        idx = self.current_content.find(marker)
        if idx != -1:
            prefix = self.current_content[: idx + len(marker)]
            return prefix + "\n  sorry\n"
        return self.current_content

    def _is_valid_line_number(self, line_no: int) -> bool:
        total = len(self.current_content.splitlines())
        return total > 0 and 1 <= line_no <= total

    def _normalize_line_number(self, line_no: int, total_lines: Optional[int] = None) -> int:
        if total_lines is None:
            total_lines = len(self.current_content.splitlines())
        if total_lines <= 0:
            return 1
        return max(1, min(line_no, total_lines))

    def _normalize_line_range(self, start_line: int, end_line: int, total_lines: int) -> Tuple[int, int]:
        if total_lines <= 0:
            return 1, 1
        start = self._normalize_line_number(start_line, total_lines)
        end = self._normalize_line_number(end_line, total_lines)
        if end < start:
            end = start
        return start, end

    def _log_state(self, step_name: str):
        """
        If log_path is set, append the current state of the code to the log file, 
        labeled with the step name.
        """
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"--- {step_name} ---\n\n")
                f.write(self.current_content)
                f.write("\n\n")

    @staticmethod
    def _byte_to_line(raw_bytes: bytes, byte_offset: int) -> int:
        """
        Converts zero-indexed byte offset to 1-indexed line number.
        """
        return raw_bytes[:byte_offset].count(b"\n") + 1

    @staticmethod
    def _strip_noop_tactics(code: str) -> str:
        """Remove standalone skip/done lines - they are no-ops and cause 'no goals' errors."""
        lines = [l for l in code.splitlines() if l.strip() not in ("skip", "done")]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _is_block_starter(line: str) -> bool:
        """
        Heuristic to identify if a line starts a new logical block, such as
        'have', 'def', etc., possibly with assignment.
        """
        stripped = line.strip()
        if stripped.startswith("_") and ":=" in stripped: return True
        if not any(stripped.startswith(cmd) for cmd in BLOCK_STARTERS): return False
        if stripped.startswith("have") and ":=" not in stripped: return False
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_sorrifier.py <path_to_lean_file>")
        sys.exit(1)
        
    target_path = sys.argv[1]
    with open(target_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    verifier = Lean4ServerScheduler(max_concurrent_requests=1, timeout=300, name="auto_sorrifier_cli")
    try:
        patcher = Sorrifier(source_code, verifier)
        fixed_code = patcher.fix_code()
        if fixed_code:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(fixed_code)
            print("Done.")
    finally:
        verifier.close()