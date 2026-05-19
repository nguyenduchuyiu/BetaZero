from __future__ import annotations

import re
import textwrap


_SORRY_RE = re.compile(r"\bsorry\b")
_PLACEHOLDER_RE = re.compile(r"\b(?:sorry|admit)\b")
_DECL_RE = re.compile(r"(?:have|let)\s+([^\s:=()]+)")


def sorry_count(code: str) -> int:
    return len(_SORRY_RE.findall(code or ""))


def placeholder_count(code: str) -> int:
    return len(_PLACEHOLDER_RE.findall(code or ""))


def verifier_placeholder_index_for_sorry(scaffold_code: str, target_index: int) -> int:
    """Map a textual `sorry` index to Lean verifier's sorry/admit index."""
    code = scaffold_code or ""
    matches = list(_SORRY_RE.finditer(code))
    if target_index < 0 or target_index >= len(matches):
        return target_index
    target_pos = matches[target_index].start()
    return sum(1 for match in _PLACEHOLDER_RE.finditer(code) if match.start() < target_pos)


def sorry_index_for_placeholder_index(code: str, placeholder_index: int) -> int | None:
    """Return textual `sorry` index for a verifier placeholder, or None for `admit`."""
    if placeholder_index < 0:
        return None

    sorry_seen = 0
    for idx, match in enumerate(_PLACEHOLDER_RE.finditer(code or "")):
        token = match.group(0)
        if idx == placeholder_index:
            return sorry_seen if token == "sorry" else None
        if token == "sorry":
            sorry_seen += 1
    return None


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
