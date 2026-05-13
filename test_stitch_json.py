import json
from betazero.search.sorrifier.stitcher import ProofStitcher

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def test_action_73():
    data = load_json('outputs/rollouts/gemini3flash/miniF2F-valid-50/aime_1983_p2.json')
    nodes = {n['id']: n for n in data['nodes']}
    
    action_73 = nodes['action_73']
    print("Action 73 Extracted Code:")
    print(action_73['extracted_lean_code'])
    print("-" * 40)
    
    # In the json, we don't have explicit children links in the node itself?
    # Actually, in BetaZero logs, AND nodes usually don't have a 'children' array in the flat list,
    # but wait, let's look at the action_73 node in python.
    print("Keys in action_73:", action_73.keys())
    if 'children' in action_73:
        child_ids = action_73['children']
    else:
        # We know from earlier inspection that state_6, state_7, state_8, state_9 are the children
        child_ids = ['state_6', 'state_7', 'state_8', 'state_9']
    
    child_proofs = []
    for cid in child_ids:
        state = nodes[cid]
        # For a state, the proof_body is either in 'content' -> 'proof_body', or we just take the proof_body field.
        # Actually, let's look at what state_6 has.
        print(f"Keys in {cid}:", state.keys())
        if 'content' in state and 'proof_body' in state['content']:
            child_proofs.append(state['content']['proof_body'])
        else:
            # Maybe we just find the solved action for this state
            # I will just extract the proof body from the state's content if available
            child_proofs.append(None)
    
    # Actually, since state_6 was SOLVED, it might have a proof_body.
    print("Child proofs found:", [bool(p) for p in child_proofs])
    
    if all(child_proofs):
        stitched = ProofStitcher.stitch(action_73['extracted_lean_code'], child_proofs)
        print("STITCHED CODE:")
        print(stitched)

if __name__ == "__main__":
    test_action_73()
