"""
AST-Based Automated Proof Patcher for Lean 4 (Bulldozer Edition)
--------------------------------------------
This module automates the process of fixing broken Lean 4 proofs by ruthlessly
replacing faulty tactics with the `sorry` axiom and deleting orphaned code.

Architecture:
1. AST-Guided Truncation: Uses Lean's AST to precisely locate tactic boundaries.
2. Boss-Hunting Fallback: Searches upward for parent blocks and clears all children.
3. Deadlock Breaker: Force-deletes lines that cause infinite loops.
"""

from __future__ import annotations
import sys
import datetime
from typing import Tuple, List, Dict, Optional, TextIO
from tqdm import tqdm

# Đã cập nhật import theo môi trường hiện tại của bạn
from betazero.env import Lean4ServerScheduler
from betazero.env.ast_parser import get_lean_ast

BLOCK_STARTERS = (
    "have", "·", ".", "cases ", "cases' ", "induction ", 
    "induction' ", "rintro ", "intro ", "calc", "match", 
    "lemma", "theorem", "def", "example"
)

TRIVIAL_TACTICS = frozenset({"skip", "done", "trivial", "decide", "rfl"})


class Sorrifier:
    def __init__(
        self, 
        repl_verifier: Lean4ServerScheduler, 
        max_cycles: int = 50,
        log_path: Optional[str] = None
    ):
        self.repl_verifier = repl_verifier
        self.max_cycles = max_cycles
        self.log_path = log_path
        self.current_content = ""
        self._last_action_msg = ""
        self._log_fp: Optional[TextIO] = None

    # ==========================================
    # LOGGING UTILS (Tích hợp từ bản mới để chạy batch)
    # ==========================================
    def _log_open(self):
        if self.log_path and self._log_fp is None:
            self._log_fp = open(self.log_path, "w", encoding="utf-8")

    def _log_close(self):
        if self._log_fp:
            self._log_fp.close()
            self._log_fp = None

    def _log(self, text: str) -> None:
        if self._log_fp:
            self._log_fp.write(text)
            if not text.endswith("\n"):
                self._log_fp.write("\n")
            self._log_fp.flush()

    # ==========================================
    # MAIN LOOP
    # ==========================================
    def fix_code(self, code: str) -> str:
        """Iteratively patch Lean 4 errors until the code compiles or max_cycles is reached."""
        self._log_open()
        try:
            self.current_content = self._strip_noop_tactics(code)
            self._last_action_msg = ""
            seen_states = set()

            with tqdm(total=self.max_cycles, desc="Processing", unit="cycle") as pbar:
                for cycle in range(1, self.max_cycles + 1):
                    try:
                        fatal_errors, unsolved_goals = self._get_lean_errors()
                    except RuntimeError as e:
                        return self._force_full_sorrify()

                    if not fatal_errors and not unsolved_goals:
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

                    # Kích hoạt Boss-Hunting nếu code không suy suyển (Oscillation)
                    if self.current_content in seen_states:
                        pbar.set_description(f"Loop detected @ L{err_line}")
                        try:
                            self._resolve_infinite_loop(err_line)
                        except IndexError as e:
                            return self._force_full_sorrify()
                        pbar.update(1)
                        continue

                    seen_states.add(self.current_content)
                    pbar.set_postfix_str(f"{'Fatal' if is_fatal else 'Unsolved'} @ L{err_line}")

                    try:
                        success = self._apply_normal_fix(err_line, is_fatal, err_msg)
                    except IndexError as e:
                        return self._force_full_sorrify()
                        
                    if not success:
                        return self.current_content
                        break

                    pbar.update(1)

        finally:
            self._log_close()

        return self.current_content

    # ==========================================
    # CORE FIXING LOGIC (Trùng khớp 100% bản cũ)
    # ==========================================
    def _resolve_infinite_loop(self, err_line: int):
        """
        Fallback resolution for correction oscillations (The Boss-Hunter).
        """
        lines = self.current_content.splitlines()
        err_line = self._normalize_line_number(err_line, total_lines=len(lines))
        original_content = self.current_content 
        
        # 1. Search backward for nearest parent block by string match
        boss_idx = -1
        for i in range(err_line - 1, -1, -1):
            line_str = lines[i].strip()
            # Keywords that start a block
            if any(line_str.startswith(kw) for kw in ["have ", "lemma ", "theorem ", "def ", "example", "·", "cases ", "match ", "induction "]):
                boss_idx = i
                break
        
        if boss_idx != -1:
            boss_line = lines[boss_idx]
            boss_indent = len(boss_line) - len(boss_line.lstrip())
            
            # 2. Replace parent block body with sorry, retain declaration
            # Handle multi-line headers: search forward for ':=' if not on this line
            found_assign = False
            if ":=" in boss_line:
                lines[boss_idx] = boss_line.split(":=")[0] + ":= by sorry"
                found_assign = True
            elif boss_line.strip().startswith("·"):
                lines[boss_idx] = " " * boss_indent + "· sorry"
                found_assign = True
            elif "=>" in boss_line:
                lines[boss_idx] = boss_line.split("=>")[0] + "=> sorry"
                found_assign = True
            else:
                # Search forward for ':='
                for j in range(boss_idx + 1, min(boss_idx + 20, len(lines))):
                    if ":=" in lines[j]:
                        lines[j] = lines[j].split(":=")[0] + ":= by sorry"
                        found_assign = True
                        # Delete lines between boss_idx and j? No, keep the header.
                        # But we should stop the child deletion from after j.
                        boss_idx = j 
                        break
            
            if not found_assign:
                # Fallback: just sorry the line where the error is
                if err_line - 1 < len(lines):
                    lines[err_line - 1] = " " * boss_indent + "sorry"
            
            # 3. Remove all child lines (greater indent) following parent
            i = boss_idx + 1
            while i < len(lines):
                if not lines[i].strip():
                    i += 1
                    continue
                curr_indent = len(lines[i]) - len(lines[i].lstrip())
                if curr_indent > boss_indent:
                    lines[i] = ""
                    i += 1
                else:
                    break
        else:
            if err_line - 1 < len(lines):
                # Try to just sorry it instead of deleting
                indent = len(lines[err_line-1]) - len(lines[err_line-1].lstrip())
                lines[err_line - 1] = " " * indent + "sorry"
            
        self.current_content = self._clean_redundant_sorries(lines)
        
        # 4. Deadlock Breaker: FORCE DELETE
        if self.current_content == original_content:
            # tqdm.write(f"Fallback didn't mutate code! Force deleting error line {err_line}.")
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
            # tqdm.write(self._last_action_msg)
            self.current_content = self._clean_redundant_sorries(lines)
            return True

        blocks = self._get_ast_lines()
        enclosing = [b for b in blocks if b["start_line"] <= error_line <= b["end_line"]]

        def emergency_fallback():
            msg = f"AST parsing failed/No valid node at L{error_line}. Applying basic single-line replacement."
            # tqdm.write(msg)
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
            
            # --- Tách mảng và xóa (Thay vì comment) ---
            new_lines = lines[:L_start - 1]
            indent = len(start_line_str) - len(start_line_str.lstrip())
            
            if is_orphan_error:
                # XÓA SẠCH
                self._last_action_msg = f"Removed orphaned tactic [{target['kind']}] L{L_start}..L{L_end}"
                new_lines.extend(lines[L_end:])
                
            elif self._is_block_starter(start_line_str) and ":=" in start_line_str:
                # XÓA BODY (Hollow out)
                self._last_action_msg = f"Hollowed out block [{target['kind']}] starting at L{L_start}"
                clean_header = start_line_str.split(":=")[0] + ":= by sorry"
                new_lines.append(clean_header)
                new_lines.extend(lines[L_end:])
                
            else:
                # THAY BẰNG SORRY
                self._last_action_msg = f"Replaced leaf tactic [{target['kind']}] L{L_start}..L{L_end}"
                new_lines.append(" " * indent + "sorry")
                new_lines.extend(lines[L_end:])
                
            # tqdm.write(self._last_action_msg)
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
            
            # Default fallback indent
            parent_indent = len(lines[L_start - 1]) - len(lines[L_start - 1].lstrip())
            indent = parent_indent + 2 
            
            for i in range(L_start, L_end): 
                line = lines[i]
                if line.strip() and not line.strip().startswith("--"):
                    indent = len(line) - len(line.lstrip())
                    break

            self._last_action_msg = f"Closed scope [{target['kind']}] at L{L_end} (Indent: {indent})"
            
            new_lines = lines[:L_end]
            new_lines.append(" " * indent + "sorry")
            new_lines.extend(lines[L_end:])
            
            self.current_content = "\n".join(new_lines) + "\n"

        self.current_content = self._clean_redundant_sorries(self.current_content.splitlines())
        return True

    # ==========================================
    # HELPERS
    # ==========================================
    def _get_lean_errors(self) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
        """Sử dụng API bất đồng bộ để không block luồng chạy song song."""
        req_ids = self.repl_verifier.submit_all_request(
            [dict(code=self.current_content)]
        )
        result = self.repl_verifier.get_all_request_outputs(req_ids)[0]

        # print(f"[REPL] verify_lean_code executed in {result.get('verify_time', 0):.4f} seconds")

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
        blocks = get_lean_ast(self.current_content)
        raw_bytes = self.current_content.encode('utf-8')
        for b in blocks:
            b["start_line"] = self._byte_to_line(raw_bytes, b["start_byte"])
            b["end_line"] = self._byte_to_line(raw_bytes, b["end_byte"])
        return blocks

    def _clean_redundant_sorries(self, lines: List[str]) -> str:
        """Keeps all lines to maintain line number stability and avoid loops."""
        return "\n".join(lines) + ("\n" if lines else "")

    def _force_full_sorrify(self) -> str:
        marker = ":= by"
        idx = self.current_content.find(marker)
        if idx != -1:
            prefix = self.current_content[: idx + len(marker)]
            return prefix + "\n  sorry\n"
        return self.current_content

    def _is_valid_line_number(self, line_no: int) -> bool:
        total = len(self.current_content.splitlines())
        return total > 0 and 1 <= line_no <= total

    def _normalize_line_number(self, line_no: int, total_lines: int | None = None) -> int:
        if total_lines is None:
            total_lines = len(self.current_content.splitlines())
        return max(1, min(line_no, total_lines)) if total_lines > 0 else 1

    @staticmethod
    def _byte_to_line(raw_bytes: bytes, byte_offset: int) -> int:
        return raw_bytes[:byte_offset].count(b"\n") + 1

    @staticmethod
    def _strip_noop_tactics(code: str) -> str:
        lines = [l for l in code.splitlines() if l.strip() not in ("skip", "done")]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _is_block_starter(line: str) -> bool:
        stripped = line.strip()
        if stripped.startswith("_") and ":=" in stripped: return True
        if not any(stripped.startswith(cmd) for cmd in BLOCK_STARTERS): return False
        if stripped.startswith("have") and ":=" not in stripped: return False
        return True


