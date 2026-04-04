"""Safe textual proof stitcher for filling skeleton subgoals."""

from __future__ import annotations
import re

class ProofStitcher:
    """Stitches child proof blocks into skeleton sorry placeholders."""

    @staticmethod
    def stitch(skeleton_code: str, child_proofs: list[str | None]) -> str:
        """
        Replaces each `sorry` in the skeleton with the corresponding child proof.
        If a child proof is None (FAILED), `sorry` remains.
        """
        # Split strictly by the word 'sorry'
        parts = re.split(r'\bsorry\b', skeleton_code)
        
        if len(parts) - 1 != len(child_proofs):
            # Fallback for LLM hallucination: mismatch between sorry count and children
            return skeleton_code

        stitched = parts[0]
        for i, proof in enumerate(child_proofs):
            if proof is not None:
                # Calculate base indentation from the line containing the 'sorry'
                lines = parts[0].splitlines()
                indent = " " * (len(lines[-1]) - len(lines[-1].lstrip())) if lines else ""
                
                # Indent child proof lines appropriately
                proof_lines = proof.splitlines()
                indented_proof = "\n".join(
                    (indent + l if idx > 0 else l) for idx, l in enumerate(proof_lines)
                )
                stitched += indented_proof
            else:
                stitched += "sorry"
                
            stitched += parts[i + 1]

        return stitched