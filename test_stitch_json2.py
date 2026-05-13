import json
import re

class ProofStitcher:
    @staticmethod
    def stitch(skeleton_code: str, child_proofs: list[str | None]) -> str:
        parts = re.split(r'\bsorry\b', skeleton_code)
        if len(parts) - 1 != len(child_proofs):
            return skeleton_code

        stitched = parts[0]
        for i, proof in enumerate(child_proofs):
            if proof is not None:
                import textwrap
                
                lines = stitched.splitlines()
                last_line = lines[-1] if lines else ""
                clean_proof = textwrap.dedent(proof).strip("\n")
                
                prefix = parts[i].rstrip()
                is_assignment = prefix.endswith(":=")
                
                if is_assignment:
                    base_indent = " " * (len(last_line) - len(last_line.lstrip()))
                    child_indent = base_indent + "  "
                    
                    if clean_proof.startswith("by ") or clean_proof.startswith("by\n"):
                        clean_proof = clean_proof[2:].lstrip()
                        
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

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

data = load_json('outputs/rollouts/gemini3flash/miniF2F-valid-50/aime_1983_p2.json')
nodes = {n['id']: n for n in data['nodes']}
action_73 = nodes['action_73']
child_ids = ['state_6', 'state_7', 'state_8', 'state_9']

child_proofs = []
for cid in child_ids:
    state = nodes[cid]
    child_proofs.append(state['content'].get('proof_body'))

print("Original action_73 extracted code:")
print(action_73['extracted_lean_code'])

stitched = ProofStitcher.stitch(action_73['extracted_lean_code'], child_proofs)
print("\n--- STITCHED CODE ---")
print(stitched)

with open('test_stitched_output.lean', 'w') as f:
    f.write(stitched)
