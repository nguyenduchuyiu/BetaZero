"""
AST-Based Automated Proof Patcher for Lean 4 (Bulldozer Edition)
--------------------------------------------
This module automates the process of fixing broken Lean 4 proofs by ruthlessly
replacing faulty tactics with the `sorry` axiom and deleting orphaned code.

Architecture:
1. AST-Guided Truncation: Uses Lean's AST to precisely locate tactic boundaries.
2. Boss-Hunting Fallback: Searches upward for parent blocks and clears all children.
3. Deadlock Breaker: Structurally prunes dependent local proof slices.
"""

from __future__ import annotations
import re
import sys
import datetime
from typing import Tuple, List, Dict, Optional, TextIO
from tqdm import tqdm

# Đã cập nhật import theo môi trường hiện tại của bạn
from gammazero.env import Lean4ServerScheduler

BLOCK_STARTERS = (
    "have", "·", ".", "cases ", "cases' ", "rcases ", "induction ", 
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

    def log(self, text: str) -> None:
        tqdm.write(text)
        self._log(text)

    def _log_diff(self, original: str, new: str) -> None:
        import difflib
        orig_lines = original.splitlines()
        new_lines = new.splitlines()
        diff = list(difflib.unified_diff(
            orig_lines, 
            new_lines, 
            fromfile="before_patch", 
            tofile="after_patch", 
            lineterm=""
        ))
        if diff:
            diff_text = "\n".join(diff)
            self.log(f"--- Code Diff (Changes made in this cycle) ---\n{diff_text}\n----------------------------------------------")

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

                    original_state = self.current_content

                    # Kích hoạt Boss-Hunting nếu code không suy suyển (Oscillation)
                    if self.current_content in seen_states:
                        pbar.set_description(f"Loop detected @ L{err_line}")
                        self.log(f"\n[LOOP DETECTED] Code oscillation detected on L{err_line}. Triggering Boss-Hunter fallback...")
                        self.log(f"--- Oscillating Code State ---\n{self.current_content}\n-----------------------------")
                        try:
                            self._resolve_infinite_loop(err_line)
                        except IndexError as e:
                            return self._force_full_sorrify()
                        
                        if self.current_content != original_state:
                            self._log_diff(original_state, self.current_content)
                            
                        pbar.update(1)
                        continue

                    seen_states.add(self.current_content)
                    pbar.set_postfix_str(f"{'Fatal' if is_fatal else 'Unsolved'} @ L{err_line}")
                    self.log(f"\n[Cycle {cycle}] Error at L{err_line} ({'Fatal' if is_fatal else 'Unsolved'}): {err_msg[:200]}")

                    try:
                        success = self._apply_normal_fix(err_line, is_fatal, err_msg)
                    except IndexError as e:
                        return self._force_full_sorrify()
                        
                    if not success:
                        return self._force_full_sorrify()

                    if self.current_content != original_state:
                        self._log_diff(original_state, self.current_content)

                    pbar.update(1)

            # Fallback to fully sorrify if loop ends without resolving all errors
            return self._force_full_sorrify()

        finally:
            self._log_close()

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
            self.log(f"[Boss-Hunter] Found parent block starter at L{boss_idx+1}: '{boss_line.strip()}'")
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
        
        # 4. Deadlock Breaker: NEVER force-delete a single line.
        # If no mutation happened, perform structural pruning instead.
        if self.current_content == original_content:
            self.log(f"[Boss-Hunter] Fallback didn't mutate code. Structural pruning at L{err_line}.")

            line = lines[err_line - 1] if 0 <= err_line - 1 < len(lines) else ""

            if self._is_local_have_line(line):
                lines = self._delete_local_decl_and_dependents(lines, err_line - 1, insert_sorry=True)
                self.current_content = self._clean_redundant_sorries(lines)
                return

            # If current line is inside a local have, find nearest enclosing have by indentation.
            curr_indent = self._indent(line) if line else 0
            for i in range(err_line - 2, -1, -1):
                if not lines[i].strip():
                    continue

                if self._indent(lines[i]) < curr_indent and self._is_local_have_line(lines[i]):
                    lines = self._delete_local_decl_and_dependents(lines, i, insert_sorry=True)
                    self.current_content = self._clean_redundant_sorries(lines)
                    return

            # Last resort: full sorrify, not single-line deletion.
            self.current_content = self._force_full_sorrify()

    def _find_innermost_non_have_tactic(self, blocks: list[dict], error_line: int) -> dict | None:
        candidates = []

        for b in blocks:
            kind = b["kind"].lower()
            if not (b["start_line"] <= error_line <= b["end_line"]):
                continue

            if "tactic" not in kind and "seq" not in kind:
                continue

            # Không chọn whole have ngay từ đầu.
            if "tactichave" in kind:
                continue

            candidates.append(b)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda b: (
                b["end_line"] - b["start_line"],
                b["end_line"],
            ),
        )

    def _replace_range_with_all_goals_sorry(
        self,
        lines: list[str],
        start_line: int,
        end_line: int,
    ) -> list[str]:
        start_idx = start_line - 1
        end_idx = end_line

        indent = self._indent(lines[start_idx])
        new_lines = lines[:start_idx]
        new_lines.append(" " * indent + "all_goals sorry")
        new_lines.extend(lines[end_idx:])

        return new_lines

    def _scope_has_sorry(self, lines: list[str], start_idx: int, end_idx_excl: int) -> bool:
        return any(
            line.strip() in {"sorry", "all_goals sorry"}
            for line in lines[start_idx:end_idx_excl]
        )

    def _hollow_have_body_with_sorry(
        self,
        lines: list[str],
        start_idx: int,
        end_idx: int,
    ) -> list[str]:
        start_line_str = lines[start_idx]
        new_lines = lines[:start_idx]
        if ":=" in start_line_str:
            clean_header = start_line_str.split(":=")[0] + ":= by sorry"
            new_lines.append(clean_header)
        else:
            indent = self._indent(start_line_str)
            new_lines.append(" " * indent + "sorry")
        new_lines.extend(lines[end_idx:])
        return new_lines

    def _apply_normal_fix(self, error_line: int, is_fatal: bool, err_msg: str) -> bool:
        lines = self.current_content.splitlines()
        error_line = self._normalize_line_number(error_line, total_lines=len(lines))

        # 1. Xử lý Trivial Tactics (Spam rác)
        line_content = lines[error_line - 1].strip()
        if line_content in TRIVIAL_TACTICS:
            lines[error_line - 1] = ""
            self._last_action_msg = f"Removed failing trivial tactic '{line_content}' at L{error_line}"
            self.log(f"[Fix] {self._last_action_msg}")
            self.current_content = self._clean_redundant_sorries(lines)
            return True

        blocks = self._get_ast_lines()
        enclosing = [b for b in blocks if b["start_line"] <= error_line <= b["end_line"]]

        def emergency_fallback():
            msg = f"AST parsing failed/No valid node at L{error_line}. Applying basic single-line replacement."
            self.log(f"[AST Fallback] {msg}")
            if is_fatal and self._is_local_have_line(lines[error_line - 1]) and self._is_header_or_elab_error(lines[error_line - 1], err_msg):
                name = self._extract_defined_name(lines[error_line - 1])
                self._last_action_msg = (
                    f"Deleted bad local declaration"
                    f"{f' `{name}`' if name else ''} and dependent sibling blocks"
                )
                self.log(f"[Fix] {self._last_action_msg}")
                pruned = self._delete_local_decl_and_dependents(lines, error_line - 1, insert_sorry=True)
                self.current_content = self._clean_redundant_sorries(pruned)
                return True
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
            indent = self._indent(start_line_str)
            
            is_orphan_error = "no goals" in err_msg.lower() or "goals accomplished" in err_msg.lower()

            # NEW 1: local have header/type error.
            # Do not produce `have bad_statement := by sorry`.
            # Delete this have and all dependent sibling blocks instead.
            if self._is_local_have_line(start_line_str) and self._is_header_or_elab_error(start_line_str, err_msg):
                name = self._extract_defined_name(start_line_str)
                self._last_action_msg = (
                    f"Deleted bad local declaration"
                    f"{f' `{name}`' if name else ''} and dependent sibling blocks"
                )
                self.log(f"[Fix] {self._last_action_msg}")
                lines = self._delete_local_decl_and_dependents(lines, L_start - 1, insert_sorry=True)
                self.current_content = self._clean_redundant_sorries(lines)
                return True

            # NEW 2: orphan/no-goals should remove a whole syntactic block, not just one line.
            if is_orphan_error:
                self._last_action_msg = f"Removed orphaned block [{target['kind']}] L{L_start}..L{L_end}"
                self.log(f"[Fix] {self._last_action_msg}")
                new_lines = lines[:L_start - 1]
                new_lines.extend(lines[L_end:])
                self.current_content = self._clean_redundant_sorries(new_lines)
                return True

            # NEW 3: branch tactic replacement must also delete following bullets.
            if self._is_branching_tactic_line(start_line_str):
                self._last_action_msg = f"Replaced branching tactic and removed bullet branches L{L_start}.."
                self.log(f"[Fix] {self._last_action_msg}")

                branch_end = self._consume_following_bullet_branches(
                    lines,
                    L_end,
                    bullet_indent=indent,
                )

                new_lines = lines[:L_start - 1]
                new_lines.append(" " * indent + "sorry")
                new_lines.extend(lines[branch_end:])
                self.current_content = self._clean_redundant_sorries(new_lines)
                return True

            # OLD behavior, but keep it only for theorem/lemma/def or safe block body.
            new_lines = lines[:L_start - 1]

            if self._is_block_starter(start_line_str) and ":=" in start_line_str:
                # For local have, only hollow body if the header itself is valid.
                # Header-error have was already handled above.
                self._last_action_msg = f"Hollowed out block [{target['kind']}] starting at L{L_start}"
                clean_header = start_line_str.split(":=")[0] + ":= by sorry"
                new_lines.append(clean_header)
                new_lines.extend(lines[L_end:])
                
            else:
                self._last_action_msg = f"Replaced leaf tactic [{target['kind']}] L{L_start}..L{L_end}"
                new_lines.append(" " * indent + "sorry")
                new_lines.extend(lines[L_end:])
                
            self.log(f"[Fix] {self._last_action_msg}")
            self.current_content = self._clean_redundant_sorries(new_lines)
                
        # 3. Xử lý Chưa chứng minh xong (Unsolved Goals)
        else:
            leaf = self._find_innermost_non_have_tactic(blocks, error_line)
            if leaf is not None:
                L_start, L_end = leaf["start_line"], leaf["end_line"]
                leaf_text = "\n".join(lines[L_start - 1:L_end])
                if "sorry" not in leaf_text:
                    self._last_action_msg = (
                        f"Replaced innermost unsolved tactic [{leaf['kind']}] "
                        f"L{L_start}..L{L_end} with all_goals sorry"
                    )
                    self.log(f"[Fix] {self._last_action_msg}")

                    new_lines = self._replace_range_with_all_goals_sorry(lines, L_start, L_end)
                    self.current_content = self._clean_redundant_sorries(new_lines)
                    return True

            scopes = ["declaration", "tactichave", "tacticcases", "tacticmatch", "tacticlet"]
            valid_nodes = [b for b in enclosing if any(s in b["kind"].lower() for s in scopes)]

            if not valid_nodes:
                valid_nodes = [b for b in enclosing if "seq" in b["kind"].lower() or "bytactic" in b["kind"].lower()]
                if not valid_nodes: return emergency_fallback()
                target = max(valid_nodes, key=lambda x: x["end_line"] - x["start_line"])
            else:
                target = min(valid_nodes, key=lambda x: x["end_line"] - x["start_line"])

            L_start, L_end = target["start_line"], target["end_line"]

            if "tactichave" in target["kind"].lower():
                if self._scope_has_sorry(lines, L_start - 1, L_end):
                    self.log(
                        f"[Fix] Unsolved have already contains sorry; hollowing have body "
                        f"L{L_start}..L{L_end}"
                    )
                    new_lines = self._hollow_have_body_with_sorry(lines, L_start - 1, L_end)
                    self.current_content = self._clean_redundant_sorries(new_lines)
                    return True

            # Prefer patching the final tactic that caused many remaining goals.
            last_idx = L_end - 1
            if 0 <= last_idx < len(lines):
                last_line = lines[last_idx]
                if self._is_multi_goal_tactic_line(last_line):
                    indent = self._indent(last_line)
                    self._last_action_msg = f"Replaced multi-goal tactic with all_goals sorry at L{L_end}"
                    self.log(f"[Fix] {self._last_action_msg}")

                    new_lines = lines[:last_idx]
                    new_lines.append(" " * indent + "all_goals sorry")
                    new_lines.extend(lines[L_end:])

                    self.current_content = self._clean_redundant_sorries(new_lines)
                    return True

            # Otherwise close every remaining goal in this tactic scope at once.
            parent_indent = self._indent(lines[L_start - 1])
            indent = parent_indent + 2

            for i in range(L_start, L_end):
                line = lines[i]
                if line.strip() and not line.strip().startswith("--"):
                    indent = self._indent(line)
                    break

            self._last_action_msg = f"Closed all remaining goals in scope [{target['kind']}] at L{L_end} (Indent: {indent})"
            self.log(f"[Fix] {self._last_action_msg}")

            new_lines = lines[:L_end]
            new_lines.append(" " * indent + "all_goals sorry")
            new_lines.extend(lines[L_end:])

            self.current_content = self._clean_redundant_sorries(new_lines)
            return True

        self.current_content = self._clean_redundant_sorries(self.current_content.splitlines())
        return True

    # ==========================================
    # HELPERS
    # ==========================================
    def _indent(self, line: str) -> int:
        return len(line) - len(line.lstrip())

    def _is_same_or_child_indent(self, lines: list[str], idx: int, base_indent: int) -> bool:
        if not lines[idx].strip():
            return True
        return self._indent(lines[idx]) >= base_indent

    def _block_end_by_indent(self, lines: list[str], start_idx: int) -> int:
        """
        Returns exclusive end index of a Lean block using indentation.
        The block includes following lines with greater indentation.
        """
        base = self._indent(lines[start_idx])
        j = start_idx + 1

        while j < len(lines):
            if not lines[j].strip():
                j += 1
                continue

            if self._indent(lines[j]) > base:
                j += 1
                continue

            break

        return j

    def _extract_defined_name(self, line: str) -> str | None:
        s = line.strip()

        # have h : P := by ...
        m = re.match(r"have\s+([A-Za-z_][A-Za-z0-9_']*)\b", s)
        if m:
            return m.group(1)

        # let x := ...
        m = re.match(r"let\s+([A-Za-z_][A-Za-z0-9_']*)\b", s)
        if m:
            return m.group(1)

        return None

    def _mentions_any_name(self, text: str, names: set[str]) -> bool:
        for name in names:
            if re.search(rf"(?<![A-Za-z0-9_']){re.escape(name)}(?![A-Za-z0-9_'])", text):
                return True
        return False

    def _is_local_have_line(self, line: str) -> bool:
        return line.strip().startswith("have ")

    def _is_header_or_elab_error(self, line: str, err_msg: str) -> bool:
        """
        True when `sorry` cannot hide the error because the error is in the
        local declaration statement/header, not in the proof body.
        """
        s = line.strip()
        msg = err_msg.lower()

        if not s.startswith("have "):
            return False

        # Strong signal: already sorrified body but same fatal error remains.
        if ":= by sorry" in s:
            return True

        header_error_markers = (
            "failed to synthesize instance",
            "type mismatch",
            "application type mismatch",
            "failed to elaborate",
            "unknown constant",
            "function expected",
            "invalid field",
            "invalid projection",
            "numerals are polymorphic",
        )

        return any(marker in msg for marker in header_error_markers)

    def _delete_local_decl_and_dependents(
        self,
        lines: list[str],
        decl_idx: int,
        insert_sorry: bool = True,
    ) -> list[str]:
        """
        Delete a bad local declaration and its following local proof slice.

        Example:
          have h_bad : BAD := by ...
          rw [h_bad]
          have h2 : ... := by ... h_bad ...
          exact ... h2 ...

        becomes:
          sorry

        This does NOT edit Lean expressions. It only prunes proof structure.
        Once the replacement `sorry` closes the current goal, following sibling
        tactics in the same local sequence would otherwise become orphaned.
        """
        base_indent = self._indent(lines[decl_idx])
        killed: set[str] = set()

        first_name = self._extract_defined_name(lines[decl_idx])
        if first_name:
            killed.add(first_name)

        delete_ranges: list[tuple[int, int]] = []

        # Delete the bad declaration block itself.
        first_end = self._block_end_by_indent(lines, decl_idx)
        delete_ranges.append((decl_idx, first_end))

        i = first_end
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue

            ind = self._indent(lines[i])

            # Leaving the local sibling scope.
            if ind < base_indent:
                break

            # Only prune sibling-level blocks. Children belong to their parent block.
            if ind > base_indent:
                i += 1
                continue

            block_end = self._block_end_by_indent(lines, i)

            defined = self._extract_defined_name(lines[i])
            if defined:
                killed.add(defined)

            delete_ranges.append((i, block_end))
            i = block_end

        out: list[str] = []
        cursor = 0

        for k, (start, end) in enumerate(delete_ranges):
            out.extend(lines[cursor:start])

            # Insert one sorry exactly at the first cut position.
            if k == 0 and insert_sorry:
                out.append(" " * base_indent + "sorry")

            cursor = end

        out.extend(lines[cursor:])
        return out

    def _consume_following_bullet_branches(
        self,
        lines: list[str],
        start_idx: int,
        bullet_indent: int,
    ) -> int:
        """
        Consume consecutive Lean bullet branches after a branching tactic.
        `start_idx` is the first line after the replaced branching tactic.
        """
        i = start_idx

        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue

            ind = self._indent(lines[i])
            s = lines[i].strip()

            if ind == bullet_indent and s.startswith("·"):
                i += 1

                # consume this branch body
                while i < len(lines):
                    if not lines[i].strip():
                        i += 1
                        continue

                    ind2 = self._indent(lines[i])
                    s2 = lines[i].strip()

                    # next sibling bullet
                    if ind2 == bullet_indent and s2.startswith("·"):
                        break

                    # left the branch group
                    if ind2 <= bullet_indent:
                        return i

                    i += 1

                continue

            break

        return i

    def _is_branching_tactic_line(self, line: str) -> bool:
        s = line.strip()
        return (
            s.startswith("rcases ")
            or s.startswith("cases ")
            or s.startswith("cases' ")
            or s.startswith("induction ")
            or s.startswith("induction' ")
        )

    def _is_multi_goal_tactic_line(self, line: str) -> bool:
        s = line.strip()
        return (
            s.startswith("interval_cases ")
            or s.startswith("cases ")
            or s.startswith("cases' ")
            or s.startswith("rcases ")
            or s.startswith("induction ")
            or s.startswith("induction' ")
            or "<;>" in s
        )

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
        from gammazero.env.ast_parser import get_lean_ast

        blocks = get_lean_ast(self.current_content)
        raw_bytes = self.current_content.encode('utf-8')
        for b in blocks:
            b["start_line"] = self._byte_to_line(raw_bytes, b["start_byte"])
            b["end_line"] = self._byte_to_line(raw_bytes, b["end_byte"])
        return blocks

    def _clean_redundant_sorries(self, lines: List[str]) -> str:
        """Collapses multiple consecutive empty lines and removes trailing ones."""
        cleaned = []
        for line in lines:
            # Keep line if it has content, or if it's the first empty line after a non-empty line
            if line.strip() or (cleaned and cleaned[-1].strip()):
                cleaned.append(line)
        
        # Remove any remaining trailing whitespace-only lines
        while cleaned and not cleaned[-1].strip():
            cleaned.pop()
            
        return "\n".join(cleaned) + ("\n" if cleaned else "")

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
