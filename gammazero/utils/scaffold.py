from __future__ import annotations

import bisect
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Iterable


_SORRY_RE = re.compile(r"\bsorry\b")
_PLACEHOLDER_RE = re.compile(r"\b(?:sorry|admit)\b")
_DECL_RE = re.compile(r"(?:have|let)\s+([^\s:=()]+)")


@dataclass(frozen=True)
class SourcePlaceholder:
    token: str
    start: int
    end: int
    line: int
    char_column: int
    byte_column: int
    end_char_column: int
    end_byte_column: int
    sorry_index: int | None


def sorry_count(code: str) -> int:
    return len(_SORRY_RE.findall(code or ""))


def _line_start_offsets(code: str) -> list[int]:
    starts = [0]
    for match in re.finditer("\n", code or ""):
        starts.append(match.end())
    return starts


def _line_number_for_offset(line_starts: list[int], offset: int) -> int:
    return bisect.bisect_right(line_starts, offset)


def _source_placeholders(code: str) -> list[SourcePlaceholder]:
    code = code or ""
    line_starts = _line_start_offsets(code)
    placeholders: list[SourcePlaceholder] = []
    sorry_seen = 0
    for match in _PLACEHOLDER_RE.finditer(code):
        line = _line_number_for_offset(line_starts, match.start())
        line_start = line_starts[line - 1]
        prefix = code[line_start : match.start()]
        token_text = match.group(0)
        token_start_col = len(prefix)
        token_end_col = token_start_col + len(token_text)
        token_start_byte_col = len(prefix.encode("utf-8"))
        token_end_byte_col = token_start_byte_col + len(token_text.encode("utf-8"))
        sorry_index = sorry_seen if token_text == "sorry" else None
        placeholders.append(
            SourcePlaceholder(
                token=token_text,
                start=match.start(),
                end=match.end(),
                line=line,
                char_column=token_start_col,
                byte_column=token_start_byte_col,
                end_char_column=token_end_col,
                end_byte_column=token_end_byte_col,
                sorry_index=sorry_index,
            )
        )
        if token_text == "sorry":
            sorry_seen += 1
    return placeholders


def _pos_line_col(pos: Any) -> tuple[int, int] | None:
    if not isinstance(pos, dict):
        return None
    try:
        return int(pos.get("line", 0)), int(pos.get("column", 0))
    except (TypeError, ValueError):
        return None


def placeholder_at_verifier_position(
    code: str,
    pos: Any,
    end_pos: Any = None,
) -> SourcePlaceholder | None:
    """Return the source placeholder reported by Lean's `pos`.

    Lean positions are stable source locations, but the meaning of `column`
    can differ across producers: some callers effectively treat it as a UTF-8
    byte column, while Python string offsets are character based.  Match both
    representations on the same source line, and use `endPos` as a tie-breaker
    when available.
    """
    line_col = _pos_line_col(pos)
    if line_col is None:
        return None
    line, column = line_col
    if line <= 0:
        return None

    candidates = [
        ph
        for ph in _source_placeholders(code)
        if ph.line == line and (ph.byte_column == column or ph.char_column == column)
    ]
    if len(candidates) == 1:
        return candidates[0]

    end_line_col = _pos_line_col(end_pos)
    if end_line_col is not None:
        end_line, end_column = end_line_col
        candidates = [
            ph
            for ph in candidates
            if ph.line == end_line
            and (ph.end_byte_column == end_column or ph.end_char_column == end_column)
        ]
        if len(candidates) == 1:
            return candidates[0]

    return None


def verifier_sorries_by_source_position(
    code: str,
    verifier_sorries: Iterable[dict[str, Any]],
    *,
    start_offset: int | None = None,
    end_offset: int | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """Pair Lean sorry records with textual `sorry` indices, sorted by source.

    The Lean REPL may report `sorries` in InfoTree traversal order, which is not
    guaranteed to match textual placeholder order.  This helper uses source
    positions instead.  `admit` records are intentionally ignored because graph
    children must correspond only to real `sorry` holes that this search node
    owns.
    """
    paired: list[tuple[int, int, dict[str, Any]]] = []
    for verifier_sorry in verifier_sorries:
        placeholder = placeholder_at_verifier_position(
            code,
            verifier_sorry.get("pos"),
            verifier_sorry.get("endPos"),
        )
        if placeholder is None or placeholder.token != "sorry" or placeholder.sorry_index is None:
            continue
        if start_offset is not None and placeholder.start < start_offset:
            continue
        if end_offset is not None and placeholder.start >= end_offset:
            continue
        paired.append((placeholder.start, placeholder.sorry_index, verifier_sorry))

    paired.sort(key=lambda item: item[0])
    return [(sorry_index, verifier_sorry) for _, sorry_index, verifier_sorry in paired]


def _line_spans(code: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    pos = 0
    for line in code.splitlines(keepends=True):
        end = pos + len(line)
        spans.append((pos, end, line))
        pos = end
    if not spans and code:
        spans.append((0, len(code), code))
    return spans


def target_decl_name(scaffold_code: str, target_index: int) -> str | None:
    """Return the local `have`/`let` name whose proof contains target `sorry`."""
    code = scaffold_code or ""
    matches = list(_SORRY_RE.finditer(code))
    if target_index < 0 or target_index >= len(matches):
        return None

    target_pos = matches[target_index].start()
    spans = _line_spans(code)
    target_line_idx = None
    target_indent = 0
    for idx, (start, end, line) in enumerate(spans):
        if start <= target_pos < end:
            target_line_idx = idx
            stripped = line.lstrip(" \t")
            target_indent = len(line) - len(stripped)
            break
    if target_line_idx is None:
        return None

    best_name: str | None = None
    for idx, (start, _, line) in enumerate(spans[: target_line_idx + 1]):
        if start > target_pos:
            break
        stripped = line.lstrip(" \t")
        if not stripped or stripped.startswith("--"):
            continue
        match = _DECL_RE.match(stripped)
        if not match:
            continue

        decl_indent = len(line) - len(stripped)
        if idx == target_line_idx or target_indent > decl_indent:
            name = match.group(1).rstrip(",")
            if name and name != "_":
                best_name = name
    return best_name


def target_subgoal_label(
    scaffold_code: str,
    target_index: int,
    *,
    target_kind: str = "",
) -> str:
    """Human-readable label for a scaffold target; used only for logging/prompts."""
    decl_name = target_decl_name(scaffold_code, target_index)
    if decl_name:
        return decl_name

    kind = (target_kind or "").strip()
    if kind == "root":
        return "root_goal"
    if kind:
        return f"{kind}_{target_index}"
    return f"target_{target_index}"


def sorry_index_after_replacement(
    scaffold_code: str,
    target_index: int,
    replacement: str,
    replacement_sorry_index: int,
) -> int:
    """Map a `sorry` inside `replacement` to its index in the patched scaffold."""
    if target_index < 0:
        return replacement_sorry_index
    return target_index + replacement_sorry_index


def _strip_optional_by(proof: str) -> str:
    proof = textwrap.dedent(proof or "").strip("\n")
    match = re.match(r"^\s*by(?P<tail>[ \t]*\n|[ \t]+|$)", proof)
    if not match:
        return proof
    return textwrap.dedent(proof[match.end():]).strip("\n")


def _format_replacement(prefix: str, replacement: str) -> str:
    clean_proof = _strip_optional_by(replacement)
    lines = prefix.splitlines()
    last_line = lines[-1] if lines else ""
    prefix_rstripped = prefix.rstrip()

    if prefix_rstripped.endswith(":="):
        base_indent = " " * (len(last_line) - len(last_line.lstrip()))
        child_indent = base_indent + "  "
        proof_lines = clean_proof.splitlines() or ["admit"]
        indented_body = "\n".join(child_indent + line for line in proof_lines)
        return ("by\n" if prefix.endswith(" ") else " by\n") + indented_body

    anchor_indent = " " * len(last_line)
    proof_lines = clean_proof.splitlines()
    if not proof_lines:
        return ""
    return "\n".join(
        (anchor_indent + line if idx > 0 else line)
        for idx, line in enumerate(proof_lines)
    )


def replace_sorry_at(scaffold_code: str, target_index: int, replacement: str) -> str:
    """Replace one textual `sorry` in a stored scaffold.

    This intentionally treats the stored scaffold as the source of truth instead
    of rebuilding a theorem from Infoview text.
    """
    parts = re.split(r"\bsorry\b", scaffold_code or "")
    count = len(parts) - 1
    if count <= 0 or target_index < 0 or target_index >= count:
        return scaffold_code

    replacement = textwrap.dedent(replacement or "").strip("\n")
    out = parts[0]
    for idx in range(count):
        out += _format_replacement(out, replacement) if idx == target_index else "sorry"
        out += parts[idx + 1]
    return out


def render_single_target_scaffold(scaffold_code: str, target_index: int) -> str:
    """Keep one `sorry` target and turn sibling sorries into `admit`."""
    parts = re.split(r"\bsorry\b", scaffold_code or "")
    count = len(parts) - 1
    if count <= 0 or target_index < 0 or target_index >= count:
        return scaffold_code

    out = parts[0]
    for idx in range(count):
        out += "sorry" if idx == target_index else _format_replacement(out, "admit")
        out += parts[idx + 1]
    return out


def isolate_sorry_target(scaffold_code: str, target_index: int) -> tuple[str, int]:
    """Store a child scaffold with exactly its target `sorry`; siblings become admits."""
    if sorry_count(scaffold_code) <= 0 or target_index < 0:
        return scaffold_code, target_index
    return render_single_target_scaffold(scaffold_code, target_index), 0
