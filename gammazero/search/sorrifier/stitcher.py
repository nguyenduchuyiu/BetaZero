"""Safe textual proof stitcher for filling skeleton subgoals."""

from __future__ import annotations
import re

class ProofStitcher:
    """Stitches child proof blocks into skeleton sorry placeholders."""

    @staticmethod
    def stitch(skeleton_code: str, child_proofs: list[str | None]) -> str:
        """Replace each `sorry` in the skeleton with the corresponding child proof.

        A `None` child means the subgoal failed; its `sorry` is preserved.
        """
        # Split strictly on the word `sorry`.
        parts = re.split(r'\bsorry\b', skeleton_code)
        
        if len(parts) - 1 != len(child_proofs):
            # Defensive fallback: skeleton/proof count mismatch (e.g. LLM hallucination).
            return skeleton_code

        stitched = parts[0]
        for i, proof in enumerate(child_proofs):
            if proof is not None:
                import textwrap
                
                lines = stitched.splitlines()
                last_line = lines[-1] if lines else ""
                
                # Normalize child indentation by removing common leading whitespace.
                clean_proof = textwrap.dedent(proof).strip("\n")
                
                # Detect whether we are filling an `:=` assignment.
                prefix = parts[i].rstrip()
                is_assignment = prefix.endswith(":=")
                
                if is_assignment:
                    base_indent = " " * (len(last_line) - len(last_line.lstrip()))
                    child_indent = base_indent + "  "
                    
                    if clean_proof.startswith("by\n"):
                        clean_proof = textwrap.dedent(clean_proof[3:]).strip("\n")
                    elif clean_proof.startswith("by "):
                        clean_proof = clean_proof[3:].strip("\n")
                        
                    proof_lines = clean_proof.splitlines()
                    indented_body = "\n".join(child_indent + l for l in proof_lines)
                    
                    if parts[i].endswith(" "):
                        indented_proof = "by\n" + indented_body
                    else:
                        indented_proof = " by\n" + indented_body
                else:
                    anchor_indent = " " * len(last_line)
                    proof_lines = clean_proof.splitlines()
                    indented_proof = "\n".join(
                        (anchor_indent + l if idx > 0 else l) for idx, l in enumerate(proof_lines)
                    )
                
                stitched += indented_proof
            else:
                stitched += "sorry"
                
            stitched += parts[i + 1]

        return stitched

    @staticmethod
    def prune_garbage(stitched_code: str, garbage_vars: list[str]) -> str:
        """Comment out every declaration line for the given garbage variables.

        Handles multi-line proof blocks by tracking indentation.
        """
        if not garbage_vars:
            return stitched_code

        lines = stitched_code.splitlines()
        out_lines = []

        # Match the exact garbage variable declarations.
        garbage_patterns = [re.compile(rf"^\s*(?:have|let)\s+{re.escape(var)}\b") for var in garbage_vars]

        skip_mode = False
        base_indent = 0  # indentation of the matched `have`/`let` line

        for line in lines:
            # 1. Detect the start of a garbage declaration.
            if any(p.search(line) for p in garbage_patterns):
                out_lines.append("-- [PRUNED] " + line)
                base_indent = len(line) - len(line.lstrip())
                skip_mode = True
                continue

            # 2. Continue pruning child lines (the proof body).
            if skip_mode:
                if not line.strip():
                    out_lines.append("-- [PRUNED] " + line)
                    continue

                curr_indent = len(line) - len(line.lstrip())
                # In Lean, the body of `have` must be indented further than the binder.
                if curr_indent > base_indent:
                    out_lines.append("-- [PRUNED] " + line)
                    continue
                else:
                    # Same-or-shallower indent ends the garbage block.
                    skip_mode = False

            # 3. Pass real code through unchanged.
            out_lines.append(line)

        return "\n".join(out_lines)
