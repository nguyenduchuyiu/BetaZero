import re
import textwrap

_LEAN_HEADER = re.compile(r"(?is)\b(theorem|lemma|example|def)\b")
_PROOF_DIVIDER = re.compile(r"(?is):=\s*by|(?<=\s)by(?=\s)")
_LEAN_FENCE = re.compile(r"```lean4\s+(.*?)\s+```", re.DOTALL | re.IGNORECASE)
_ESCAPED_WS = re.compile(r"(?:\\[ \t\r\n\f\v])+")


TRUNCATED_THINK_FEEDBACK = (
    "OUTPUT TRUNCATED: the response did not end with a complete final "
    "```lean4 ... ``` code block. The thinking process is too long; output is "
    "likely truncated. Think shorter, then finish with exactly one final Lean "
    "code block and no text after it."
)

SUBGOAL_EXTRACTION_FEEDBACK = (
    "SUBGOAL EXTRACTION FAILURE: a final ```lean4 ... ``` block was found, but "
    "the verifier could not extract the replacement for the unique target "
    "`sorry`. Reproduce the parent scaffold exactly, replace only the unique "
    "`sorry` target, and keep sibling `admit` placeholders unchanged."
)

EMPTY_LEAN_CODE_FEEDBACK = (
    "NO EXTRACTABLE LEAN CODE: finish with exactly one final ```lean4 ... ``` "
    "block containing the Lean answer, and do not add text after the code block."
)

INVALID_SKELETON_FEEDBACK = (
    "SKELETON POLICY VIOLATION: a skeleton replacement must not contain `admit`, "
    "and every `sorry` must be a named leaf obligation of the form "
    "`have h_name : proposition := sorry` or `have h_name : proposition := by sorry`. "
    "Do not leave a naked final `sorry`; close the target assembly using the named leaves."
)


def strip_lean_comments(code: str) -> str:
    return re.sub(r"/-(?:.|\n)*?-/|--.*", "", code)


def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    if end == -1:
        end = len(text)
    return start, end


def _sorry_is_named_have_leaf(clean_code: str, match: re.Match[str]) -> bool:
    line_start, line_end = _line_bounds(clean_code, match.start())
    line = clean_code[line_start:line_end]
    after = line[match.end() - line_start :]

    if after.strip():
        return False

    before = clean_code[:match.start()].rstrip()
    return bool(
        re.search(
            r"\b(have|let)\s+[A-Za-z_][A-Za-z0-9_']*\b[\s\S]*?:=\s*(?:by\s*)?$",
            before,
        )
    )


def validate_skeleton_replacement(code: str) -> str:
    """Return a policy feedback string if a skeleton body is structurally invalid."""
    clean = strip_lean_comments(code or "")
    if re.search(r"\badmit\b", clean):
        return INVALID_SKELETON_FEEDBACK

    for match in re.finditer(r"\bsorry\b", clean):
        if not _sorry_is_named_have_leaf(clean, match):
            return INVALID_SKELETON_FEEDBACK
    return ""


def _final_lean_block(raw: str) -> str:
    t = raw.strip()

    # Reject ChatML control tokens.
    if "<|im_" in t:
        return ""

    # The final ```lean4``` block must close at the very end of the response.
    # Earlier draft fences inside a thinking section are not the answer.
    fences = list(_LEAN_FENCE.finditer(t))
    if not fences or fences[-1].end() != len(t):
        return ""

    code_block = fences[-1].group(1)

    # Strip comments before checking for `...` placeholders.
    code_no_comments = strip_lean_comments(code_block)

    # Reject only standalone `...` lines (a lazy-model placeholder),
    # not `...` that may legitimately appear inside other syntax/comments.
    if re.search(r"^\s*\.\.\.\s*$", code_no_comments, re.MULTILINE):
        return ""

    return code_block


def _is_length_finish_reason(finish_reason: object) -> bool:
    if finish_reason is None:
        return False
    reason = str(finish_reason).lower()
    return any(token in reason for token in ("max_token", "max_tokens", "length"))


def explain_empty_lean_code(
    raw: str,
    *,
    subgoal: bool = False,
    finish_reason: object = None,
) -> str:
    """Explain why extraction returned an empty Lean code string for logs."""
    if _is_length_finish_reason(finish_reason):
        return TRUNCATED_THINK_FEEDBACK

    t = (raw or "").strip()
    if not t:
        return EMPTY_LEAN_CODE_FEEDBACK

    if "<|im_" in t:
        return (
            "NO EXTRACTABLE LEAN CODE: the response contains ChatML control "
            "tokens. Output only the assistant content, ending with one final "
            "```lean4 ... ``` block."
        )

    fences = list(_LEAN_FENCE.finditer(t))
    if not fences:
        if "```lean4" in t or "```" in t or "<think>" in t:
            return TRUNCATED_THINK_FEEDBACK
        return EMPTY_LEAN_CODE_FEEDBACK

    if fences[-1].end() != len(t):
        return TRUNCATED_THINK_FEEDBACK

    code_block = fences[-1].group(1)
    code_no_comments = strip_lean_comments(code_block)
    if re.search(r"^\s*\.\.\.\s*$", code_no_comments, re.MULTILINE):
        return (
            "NO EXTRACTABLE LEAN CODE: the final Lean block contains a standalone "
            "`...` placeholder. Replace placeholders with actual Lean code."
        )

    if subgoal:
        return SUBGOAL_EXTRACTION_FEEDBACK

    if not _LEAN_HEADER.search(code_block) or not _PROOF_DIVIDER.search(code_block):
        return (
            "NO EXTRACTABLE LEAN CODE: the final Lean block is not a complete "
            "Lean declaration with a `:= by` proof."
        )

    return EMPTY_LEAN_CODE_FEEDBACK


def get_lean_code(raw: str, *, allow_body: bool = False) -> str:
    """Return the proof body when the response is a fully valid Lean action.

    Preserves the original whitespace and indentation of the proof body.
    """
    code_block = _final_lean_block(raw)
    if not code_block:
        return ""
    code_no_comments = strip_lean_comments(code_block)

    # Require a header and a proof divider unless the caller accepts a raw
    # tactic body.
    if not _LEAN_HEADER.search(code_block) or not _PROOF_DIVIDER.search(code_block):
        if not allow_body:
            return ""
        body_clean = code_no_comments.strip()
        if not body_clean or body_clean == "sorry" or "<|im_" in code_block:
            return ""
        return textwrap.dedent(code_block).strip("\n")

    divider_match = _PROOF_DIVIDER.search(code_block)

    # Take everything after `by`, preserving newlines and indentation.
    proof_body = code_block[divider_match.end():]

    # Reject bodies that are empty or contain only comments/`sorry`.
    body_clean = re.sub(r"/-(?:.|\n)*?-/|--.*", "", proof_body).strip()
    if not body_clean or body_clean == "sorry" or "<|im_" in proof_body:
        return ""

    return proof_body


def _strip_leading_by(proof: str) -> str:
    """Return the tactic body inside an optional leading `by` wrapper."""
    proof = textwrap.dedent(proof).strip("\n")
    match = re.match(r"^\s*by(?P<tail>[ \t]*\n|[ \t]+|$)", proof)
    if not match:
        return proof

    body = proof[match.end():]
    # The previous regex used `\s+`, which consumed the indentation of the first
    # tactic line after `by\n` and turned aligned tactic blocks into accidental
    # nested blocks when stitched back into the scaffold.
    return textwrap.dedent(body).strip("\n")


def _lean_fragment_pattern(fragment: str) -> str:
    """Escape Lean text while allowing harmless whitespace layout changes."""
    return _ESCAPED_WS.sub(r"\\s+", re.escape(fragment))


def _target_decl_name(skeleton_code: str, target_child_index: int) -> str | None:
    matches = list(re.finditer(r"\bsorry\b", skeleton_code or ""))
    if target_child_index < 0 or target_child_index >= len(matches):
        return None
    prefix = skeleton_code[: matches[target_child_index].start()]
    decl_matches = list(
        re.finditer(
            r"(?:^|\n)\s*(?:have|let)\s+([A-Za-z_][A-Za-z0-9_']*)\b",
            prefix,
        )
    )
    if not decl_matches:
        return None
    return decl_matches[-1].group(1)


def _extract_decl_replacement(proof_body: str, decl_name: str) -> str:
    """Extract `decl_name`'s assigned proof body from a partial parent scaffold."""
    lines = proof_body.splitlines()
    start = None
    start_indent = 0
    decl_pattern = re.compile(rf"(?:have|let)\s+{re.escape(decl_name)}\b")

    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if decl_pattern.match(stripped):
            start = idx
            start_indent = len(line) - len(stripped)
            break

    if start is None:
        return ""

    end = start + 1
    while end < len(lines):
        line = lines[end]
        stripped = line.lstrip()
        if stripped and len(line) - len(stripped) <= start_indent:
            break
        end += 1

    block = "\n".join(lines[start:end])
    assign = re.search(r":=", block)
    if not assign:
        return ""
    return _strip_leading_by(block[assign.end():])


def _get_subgoal_code_by_decl(
    proof_body: str,
    skeleton_code: str,
    target_child_index: int,
) -> str:
    decl_name = _target_decl_name(skeleton_code, target_child_index)
    if decl_name is None:
        return ""

    replacement = _extract_decl_replacement(proof_body, decl_name)
    replacement_clean = strip_lean_comments(replacement).strip()
    if not replacement_clean or replacement_clean in {"sorry", "admit"}:
        return ""
    return replacement


def get_subgoal_tactic_code(
    raw: str,
    skeleton_code: str,
    target_child_index: int,
    *,
    allow_partial_scaffold: bool = False,
) -> str:
    """Extract only the replacement proof for one skeleton `sorry`.

    The subgoal tactic prompt asks the model to return the whole parent scaffold
    so Lean elaborates coercions in the same context. Search still stores and
    stitches only the proof replacing the target placeholder.
    """
    code_block = _final_lean_block(raw)
    if not code_block:
        return ""

    # Backward-compatible fallback for older body-only responses.
    if not _LEAN_HEADER.search(code_block) or not _PROOF_DIVIDER.search(code_block):
        return get_lean_code(raw, allow_body=True)

    divider_match = _PROOF_DIVIDER.search(code_block)
    if not divider_match:
        return ""

    proof_body = textwrap.dedent(code_block[divider_match.end():]).strip("\n")
    skeleton_body = textwrap.dedent(skeleton_code or "").strip("\n")
    parts = re.split(r"\bsorry\b", skeleton_body)
    sorry_count = len(parts) - 1
    if sorry_count <= 0 or target_child_index < 0 or target_child_index >= sorry_count:
        return ""

    pattern = _lean_fragment_pattern(parts[0])
    for i in range(sorry_count):
        if i == target_child_index:
            pattern += r"(?P<replacement>.*?)"
        else:
            pattern += r"(?:by\s+admit|admit)"
        pattern += _lean_fragment_pattern(parts[i + 1])

    match = re.search(pattern, proof_body, flags=re.DOTALL)
    if not match:
        if allow_partial_scaffold:
            return _get_subgoal_code_by_decl(
                proof_body,
                skeleton_body,
                target_child_index,
            )
        return ""

    replacement = _strip_leading_by(match.group("replacement"))
    replacement_clean = strip_lean_comments(replacement).strip()
    if not replacement_clean or replacement_clean in {"sorry", "admit"}:
        return ""
    return replacement


def get_subgoal_skeleton_code(
    raw: str,
    skeleton_code: str,
    target_child_index: int,
    *,
    allow_partial_scaffold: bool = False,
) -> str:
    """Extract the mini-skeleton replacing one parent-skeleton `sorry`.

    Unlike tactic extraction, a skeleton replacement is allowed to contain new
    `sorry` leaves. Sibling placeholders in the returned parent scaffold must
    remain as `admit`.
    """
    code_block = _final_lean_block(raw)
    if not code_block:
        return ""

    if not _LEAN_HEADER.search(code_block) or not _PROOF_DIVIDER.search(code_block):
        return get_lean_code(raw, allow_body=True)

    divider_match = _PROOF_DIVIDER.search(code_block)
    if not divider_match:
        return ""

    proof_body = textwrap.dedent(code_block[divider_match.end():]).strip("\n")
    skeleton_body = textwrap.dedent(skeleton_code or "").strip("\n")
    parts = re.split(r"\bsorry\b", skeleton_body)
    sorry_count = len(parts) - 1
    if sorry_count <= 0 or target_child_index < 0 or target_child_index >= sorry_count:
        return ""

    pattern = _lean_fragment_pattern(parts[0])
    for i in range(sorry_count):
        if i == target_child_index:
            pattern += r"(?P<replacement>.*?)"
        else:
            pattern += r"(?:by\s+admit|admit)"
        pattern += _lean_fragment_pattern(parts[i + 1])

    match = re.search(pattern, proof_body, flags=re.DOTALL)
    if not match:
        if allow_partial_scaffold:
            return _get_subgoal_code_by_decl(
                proof_body,
                skeleton_body,
                target_child_index,
            )
        return ""

    replacement = _strip_leading_by(match.group("replacement"))
    replacement_clean = strip_lean_comments(replacement).strip()
    if not replacement_clean or replacement_clean == "admit":
        return ""
    return replacement
