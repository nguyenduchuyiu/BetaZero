"""
AST-Based Automated Proof Patcher for Lean 4
--------------------------------------------
This module automates the process of fixing broken Lean 4 proofs by replacing 
faulty tactics with the `sorry` axiom. 

Architecture:
1. AST-Guided Truncation: Uses Lean's AST (with Elaborator) to precisely locate tactic boundaries using Spatial Heuristics (Byte Length).
2. Indentation Heuristics: Infers structural hierarchy where AST lacks context (e.g., closing scopes).
3. Oscillation Fallback: Detects infinite correction loops caused by Lean's syntax 
   intolerance and resets the parent block to prevent halting.
"""

from __future__ import annotations
import sys
import datetime
from typing import Tuple, List, Dict, TextIO, Optional
from tqdm import tqdm
from betazero.env.lean_verifier import Lean4ServerScheduler
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
        log_path: Optional[str] = None,
    ):
        self.repl_verifier = repl_verifier
        self.max_cycles = max_cycles
        self.log_path = log_path
        self.current_content = ""
        self._last_action_msg = ""
        self._log_fp: Optional[TextIO] = None

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

    def _log_section(self, title: str) -> None:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        bar = "=" * 76
        self._log(f"\n{bar}\n[{ts}] {title}\n{bar}\n")

    @staticmethod
    def _format_numbered_source(content: str, width: int = 5) -> str:
        lines = content.splitlines()
        out = [f"{i:>{width}} | {line}" for i, line in enumerate(lines, start=1)]
        return "\n".join(out) + ("\n" if out else "")

    def _log_source_block(self, label: str, content: str, err_line: int | None = None) -> None:
        self._log(f"--- {label} (numbered) ---")
        if err_line is not None:
            self._log(f"(error line ref: L{err_line})")
        self._log(self._format_numbered_source(content))

    def _log_error_batch(
        self,
        fatal_errors: List[Tuple[int, str]],
        unsolved_goals: List[Tuple[int, str]],
        primary_line: int,
        primary_msg: str,
        is_fatal: bool,
    ) -> None:
        self._log("Primary (first) issue:")
        self._log(f"  kind: {'fatal' if is_fatal else 'unsolved_goals'}")
        self._log(f"  line: {primary_line}")
        self._log(f"  message: {primary_msg}")
        if fatal_errors:
            self._log("All fatal errors (line, message):")
            for ln, msg in fatal_errors:
                self._log(f"  L{ln}: {msg[:500]}{'…' if len(msg) > 500 else ''}")
        if unsolved_goals:
            self._log("All unsolved goals (line, message):")
            for ln, msg in unsolved_goals:
                self._log(f"  L{ln}: {msg[:500]}{'…' if len(msg) > 500 else ''}")

    def fix_code(self, code: str) -> str:
        """Iteratively patch Lean 4 errors until the code compiles or max_cycles is reached."""
        self._log_open()
        try:
            self.current_content = self._strip_noop_tactics(code)
            self._last_action_msg = ""
            seen_states = set()

            if self._log_fp:
                self._log_section("Sorrifier start")
                self._log("After _strip_noop_tactics, initial source:")
                self._log_source_block("INITIAL", self.current_content)

            with tqdm(total=self.max_cycles, desc="Processing", unit="cycle") as pbar:
                for cycle in range(1, self.max_cycles + 1):
                    try:
                        fatal_errors, unsolved_goals = self._get_lean_errors()
                    except RuntimeError as e:
                        tqdm.write(f"\nHALTED: {e}")
                        if self._log_fp:
                            self._log_section(f"HALTED (cycle {cycle}) — Lean/runtime")
                            self._log(str(e))
                            self._log_source_block("STATE AT HALT", self.current_content)
                        return self._force_full_sorrify()

                    if not fatal_errors and not unsolved_goals:
                        if self._log_fp:
                            self._log_section(f"SUCCESS (cycle {cycle})")
                            self._log("No fatal errors and no unsolved goals.")
                            self._log_source_block("FINAL", self.current_content)
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
                        if self._log_fp:
                            self._log_section(f"Cycle {cycle} — OSCILLATION / parent block reset")
                            self._log_error_batch(fatal_errors, unsolved_goals, err_line, err_msg, is_fatal)
                            self._log_source_block("BEFORE reset", self.current_content, err_line)
                        try:
                            self._resolve_infinite_loop(err_line)
                        except IndexError as e:
                            tqdm.write(f"Index error during oscillation fallback: {e}. Force full sorrify.")
                            if self._log_fp:
                                self._log_section("FORCE full sorrify (IndexError in oscillation)")
                                self._log(str(e))
                                self._log_source_block("STATE", self.current_content)
                            return self._force_full_sorrify()
                        if self._log_fp:
                            self._log("--- AFTER reset ---")
                            if self._last_action_msg:
                                self._log(f"action: {self._last_action_msg}")
                            self._log_source_block("AFTER reset", self.current_content, err_line)
                        pbar.update(1)
                        continue

                    seen_states.add(self.current_content)
                    pbar.set_postfix_str(f"{'Fatal' if is_fatal else 'Unsolved'} @ L{err_line}")

                    if self._log_fp:
                        self._log_section(f"Cycle {cycle} — normal fix")
                        self._log_error_batch(fatal_errors, unsolved_goals, err_line, err_msg, is_fatal)
                        self._log_source_block("BEFORE fix", self.current_content, err_line)

                    try:
                        success = self._apply_normal_fix(err_line, is_fatal, err_msg)
                    except IndexError as e:
                        tqdm.write(f"Index error during normal fix: {e}. Force full sorrify.")
                        if self._log_fp:
                            self._log_section("FORCE full sorrify (IndexError in normal fix)")
                            self._log(str(e))
                            self._log_source_block("STATE", self.current_content)
                        return self._force_full_sorrify()
                    if not success:
                        tqdm.write(f"\nHALTED: Unrecoverable error at line {err_line}.")
                        if self._log_fp:
                            self._log_section(f"HALTED (cycle {cycle}) — unrecoverable")
                            self._log(f"line {err_line}")
                            self._log_source_block("STATE AT HALT", self.current_content, err_line)
                        break

                    if self._log_fp:
                        self._log("--- AFTER fix ---")
                        if self._last_action_msg:
                            self._log(f"action: {self._last_action_msg}")
                        self._log_source_block("AFTER fix", self.current_content, err_line)

                    pbar.update(1)

            if self._log_fp:
                self._log_section("Run finished — loop ended without early success return")
                self._log("(Either max_cycles exhausted or break after unrecoverable error.)")
                self._log_source_block("FINAL", self.current_content)
        finally:
            self._log_close()

        return self.current_content

    # ==========================================
    # CORE FIXING LOGIC
    # ==========================================

    def _resolve_infinite_loop(self, err_line: int):
        lines = self.current_content.splitlines()
        err_line = self._normalize_line_number(err_line, total_lines=len(lines))
        
        line_str = lines[err_line - 1]
        indent = len(line_str) - len(line_str.lstrip())
        
        if any(line_str.strip().startswith(kw) for kw in ["lemma", "theorem", "def", "example"]):
            lines.append(" " * 2 + "sorry")
        else:
            # Phanh khẩn cấp xịn: Comment dòng lỗi và toàn bộ các dòng con thụt lề sâu hơn
            lines[err_line - 1] = "-- " + lines[err_line - 1]
            i = err_line
            while i < len(lines) and lines[i].strip():
                curr_indent = len(lines[i]) - len(lines[i].lstrip())
                if curr_indent > indent:
                    lines[i] = "-- " + lines[i]
                    i += 1
                else:
                    break
            # Nhét sorry vào cuối để thoát kẹt
            lines.insert(i, " " * indent + "sorry")
            
        self.current_content = "\n".join(lines) + "\n"

    def _apply_normal_fix(self, error_line: int, is_fatal: bool, err_msg: str) -> bool:
        lines = self.current_content.splitlines()
        error_line = self._normalize_line_number(error_line, total_lines=len(lines))

        line_content = lines[error_line - 1].strip()
        if line_content in TRIVIAL_TACTICS:
            lines[error_line - 1] = ""
            self._last_action_msg = f"Removed failing trivial tactic '{line_content}' at L{error_line}"
            tqdm.write(self._last_action_msg)
            self.current_content = "\n".join(lines) + "\n"
            return True

        blocks = self._get_ast_lines()
        enclosing = [b for b in blocks if b["start_line"] <= error_line <= b["end_line"] and b["kind"] != "Module"]

        def emergency_fallback():
            indent = len(lines[error_line - 1]) - len(lines[error_line - 1].lstrip())
            line_str = lines[error_line - 1]
            if any(line_str.strip().startswith(kw) for kw in ["lemma", "theorem", "def", "example"]):
                lines.append(" " * 2 + "sorry")
            else:
                lines[error_line - 1] = "-- " + line_str + "\n" + " " * indent + "sorry"
            self.current_content = "\n".join(lines) + "\n"
            return True

        if not enclosing:
            return emergency_fallback()

        if is_fatal:
            # CHỐNG BÃO SORRY: Lean đòi Command, tuyệt đối không chèn sorry bừa bãi
            if "expected command" in err_msg.lower():
                indent = len(lines[error_line - 1]) - len(lines[error_line - 1].lstrip())
                lines[error_line - 1] = "-- " + lines[error_line - 1]
                i = error_line
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("--"):
                    curr_indent = len(lines[i]) - len(lines[i].lstrip())
                    if curr_indent > indent:
                        lines[i] = "-- " + lines[i]
                        i += 1
                    else:
                        break
                self._last_action_msg = f"Commented invalid syntax block at L{error_line}"
                self.current_content = "\n".join(lines) + "\n"
                return True

            valid_nodes = [b for b in enclosing if "command" not in b["kind"].lower()]
            if not valid_nodes: return emergency_fallback()
            
            target = min(valid_nodes, key=lambda x: x["end_byte"] - x["start_byte"])
            L_start, L_end = target["start_line"], target["end_line"]
            start_line_str = lines[L_start - 1]
            indent = len(start_line_str) - len(start_line_str.lstrip())
            
            is_orphan_error = "no goals" in err_msg.lower() or "goals accomplished" in err_msg.lower()
            
            if is_orphan_error:
                for i in range(L_start - 1, L_end):
                    lines[i] = "-- " + lines[i]
                self._last_action_msg = f"Commented orphaned tactic [{target['kind']}] L{L_start}..L{L_end}"
                self.current_content = "\n".join(lines) + "\n"
                return True

            if self._is_block_starter(start_line_str):
                new_lines = lines[:L_end]
                new_lines.append(" " * (indent + 2) + "sorry")
                new_lines.extend(lines[L_end:])
                self._last_action_msg = f"Appended sorry to block [{target['kind']}] starting at L{L_start}"
                self.current_content = "\n".join(new_lines) + "\n"
            else:
                for i in range(L_start - 1, L_end):
                    lines[i] = "-- " + lines[i]
                new_lines = lines[:L_end]
                new_lines.append(" " * indent + "sorry")
                new_lines.extend(lines[L_end:])
                self._last_action_msg = f"Commented failing tactic [{target['kind']}] L{L_start}..L{L_end}"
                self.current_content = "\n".join(new_lines) + "\n"

        else: 
            # UNSOLVED GOALS: Đã fix danh sách scope chuẩn, đéo bao giờ vồ nhầm lá nữa!
            scopes = [
                "tactichave__", "tacticcases__", "tacticlet__", 
                "tacticinduction__", "tacticcalc__", "tacticmatch__", 
                "bytactic", "declval"
            ]
            valid_nodes = [b for b in enclosing if any(s in b["kind"].lower() for s in scopes)]
            
            if not valid_nodes:
                lines.append("  sorry")
                self.current_content = "\n".join(lines) + "\n"
                return True
                
            target = min(valid_nodes, key=lambda x: x["end_byte"] - x["start_byte"])
            L_start, L_end = target["start_line"], target["end_line"]
            
            parent_indent = len(lines[L_start - 1]) - len(lines[L_start - 1].lstrip())
            indent = parent_indent + 2 
            
            for i in range(L_start, L_end):
                line = lines[i]
                if line.strip() and not line.strip().startswith("--"):
                    indent = len(line) - len(line.lstrip())
                    break

            self._last_action_msg = f"Closed scope [{target['kind']}] at L{L_end} (Indent: {indent})"
            tqdm.write(self._last_action_msg)
            
            new_lines = lines[:L_end]
            new_lines.append(" " * indent + "sorry")
            new_lines.extend(lines[L_end:])
            
            self.current_content = "\n".join(new_lines) + "\n"

        return True

    def _get_lean_errors(self) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
        result = self.repl_verifier.verify(self.current_content)

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
        cleaned = []
        for line in lines:
            if line == "": continue
            stripped = line.strip()
            if stripped == "sorry" and cleaned and cleaned[-1].strip() == "sorry": continue
            cleaned.append(line)
        return "\n".join(cleaned) + "\n"

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
        if total_lines is None: total_lines = len(self.current_content.splitlines())
        if total_lines <= 0: return 1
        return max(1, min(line_no, total_lines))

    def _normalize_line_range(self, start_line: int, end_line: int, total_lines: int) -> Tuple[int, int]:
        if total_lines <= 0: return 1, 1
        start = self._normalize_line_number(start_line, total_lines)
        end = self._normalize_line_number(end_line, total_lines)
        if end < start: end = start
        return start, end

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