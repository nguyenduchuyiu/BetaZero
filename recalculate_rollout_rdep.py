import argparse
import json
import os
import re
from pathlib import Path
from tqdm import tqdm

from betazero.core.nodes import ProofState, Action
from betazero.env.lean_env import LeanEnv
from betazero.search.reward.calculator import RewardCalculator
from betazero.search.reward.reward_assigner import DependencyRewardAssigner
from betazero.search.sorrifier.stitcher import ProofStitcher
from betazero.env.lean_env import Lean4ServerScheduler

def process_file(filepath: Path, lean: LeanEnv, reward_calc: RewardCalculator, assigner: DependencyRewardAssigner):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return

    nodes = {n['id']: n for n in data.get('nodes', [])}
    edges = data.get('edges', [])
    
    # Build relationships
    action_to_children = {}
    state_to_actions = {}
    action_to_parent = {}
    
    for edge in edges:
        src = edge['source']
        tgt = edge['target']
        rel = edge.get('relation')
        
        if rel == "expanded_to":
            # src is state, tgt is action
            state_to_actions.setdefault(src, []).append(tgt)
            action_to_parent[tgt] = src
        elif rel == "subgoal":
            # src is action, tgt is state
            action_to_children.setdefault(src, []).append(tgt)

    # 0. Refresh extracted_lean_code if missing or needed
    from betazero.policy.output_parser import get_lean_code
    for node_id, node in nodes.items():
        if node.get('type') == "AND":
            raw_content = node.get('content', '')
            # If raw_content is a dict, it's a state node or already processed. 
            # Actions usually have 'content' as a string (the raw LLM response).
            if isinstance(raw_content, str) and raw_content:
                new_extracted = get_lean_code(raw_content)
                if new_extracted:
                    node['extracted_lean_code'] = new_extracted

    # 1. Recalculate r_dep for all skeletons
    updated_count = 0
    for node_id, node in nodes.items():
        if node.get('type') == "AND" and node.get('action_type') == "skeleton":
            extracted_code = node.get('extracted_lean_code', '')
            if not extracted_code: continue
            children_ids = action_to_children.get(node_id, [])
            if not children_ids: continue
            
            parent_id = action_to_parent.get(node_id)
            if not parent_id: continue
            parent_state_data = nodes[parent_id]
            
            prompt = node.get('prompt', '')
            header = "open BigOperators Nat Real Topology\n"
            match = re.search(r"\[PROBLEM\]\n```lean4\n(.*?)\ntheorem", prompt, re.DOTALL)
            if match: header = match.group(1).strip() + "\n"
                
            # Use current proofs from JSON (might be old, but we will refresh them in order)
            child_proofs = [nodes[cid].get('content', {}).get('proof_body', '  sorry') for cid in children_ids]
            stitched_code = ProofStitcher.stitch(extracted_code, child_proofs)
            
            full_match = re.search(r"\[PROBLEM\]\n```lean4\n(.*?:= by\n)  sorry\n```", prompt, re.DOTALL)
            if full_match: full_code = full_match.group(1) + stitched_code
            else:
                from betazero.utils.lean_cmd import build_theorem
                ps = ProofState(
                    goal=parent_state_data['content']['goal'], 
                    context=parent_state_data['content'].get('context', ''),
                    header=header
                )
                full_code = build_theorem(ps, stitched_code)
                
            allowed_vars = assigner._extract_sorry_vars(extracted_code)
            dep_analysis = lean.analyze_dependencies(full_code, allowed_vars=allowed_vars)
            
            r_dep_score = 0.0 if len(dep_analysis.get("core_failed", [])) > 0 else reward_calc.r_dep({
                "core": dep_analysis.get("core_solved", []),
                "benign": dep_analysis.get("benign", []),
                "malignant": dep_analysis.get("malignant", [])
            })
            
            node.setdefault('metrics', {})['r_dep'] = r_dep_score
            updated_count += 1

    # 2. Refresh proof_body for all states (OR nodes) bottom-up or via recursion
    memo_proofs = {}

    def get_proof_for_action(a_id):
        a_node = nodes[a_id]
        if a_node['action_type'] == "tactic":
            return a_node['extracted_lean_code']
        
        # Skeleton: stitch child proofs
        c_ids = action_to_children.get(a_id, [])
        c_proofs = [get_proof_for_state(c) for c in c_ids]
        return ProofStitcher.stitch(a_node['extracted_lean_code'], c_proofs)

    def get_proof_for_state(s_id):
        if s_id in memo_proofs: return memo_proofs[s_id]
        s_node = nodes[s_id]
        if s_node['status'] != "SOLVED":
            res = "  sorry"
        else:
            # Pick the best solved action (BetaZero picks first successful one for logger)
            best_action = next((a for a in state_to_actions.get(s_id, []) if nodes[a]['status'] == "SOLVED"), None)
            res = get_proof_for_action(best_action) if best_action else "  sorry"
        
        memo_proofs[s_id] = res
        return res

    for s_id, s_node in nodes.items():
        if s_node.get('type') == "OR" and s_node['status'] == "SOLVED":
            new_pb = get_proof_for_state(s_id)
            if s_node['content'].get('proof_body') != new_pb:
                s_node['content']['proof_body'] = new_pb
                updated_count += 1
            
    # 3. Propagate rewards (Backup) to update Q_value and V_value
    # Use standard BetaZero backup: V(s) = max(Q(a)), Q(a) = r_env + solved * (r_dep + gamma * min(V(children)))
    gamma = 1.0
    v_cache = {}
    q_cache = {}
    visiting_v = set()
    visiting_q = set()

    def get_v(s_id):
        if s_id in v_cache: return v_cache[s_id]
        if s_id in visiting_v: return 0.0
        visiting_v.add(s_id)
        
        child_actions = state_to_actions.get(s_id, [])
        if not child_actions:
            res = 0.0
        else:
            res = max(get_q(a) for a in child_actions)
            
        visiting_v.remove(s_id)
        v_cache[s_id] = res
        return res

    def get_q(a_id):
        if a_id in q_cache: return q_cache[a_id]
        if a_id in visiting_q: return 0.0
        visiting_q.add(a_id)
        
        a_node = nodes[a_id]
        r_env = a_node['metrics'].get('r_env', 0.0)
        solved = 1.0 if a_node['status'] == "SOLVED" else 0.0
        
        if a_node['action_type'] == "tactic":
            # For tactics, Q = r_env + solved
            res = r_env + solved
        else:
            # For skeletons, Q = r_env + solved * (r_dep + gamma * min(V_children))
            r_dep = a_node['metrics'].get('r_dep', 0.0)
            c_ids = action_to_children.get(a_id, [])
            if not c_ids:
                future = 0.0
            else:
                future = gamma * min(get_v(c) for c in c_ids)
            res = r_env + solved * (r_dep + future)
            
        visiting_q.remove(a_id)
        q_cache[a_id] = res
        return res

    # Bắt đầu backup từ root
    root_id = data.get('root_id')
    if root_id:
        get_v(root_id)
        
        # Cập nhật lại metrics trong nodes
        for a_id, q_val in q_cache.items():
            nodes[a_id].setdefault('metrics', {})['Q_value'] = q_val
        for s_id, v_val in v_cache.items():
            nodes[s_id].setdefault('metrics', {})['V_value'] = v_val
            
    if updated_count > 0 or len(q_cache) > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # print(f"Updated {updated_count} nodes and recalculated Q/V in {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Recalculate offline r_dep for rollout JSON files")
    parser.add_argument("--json-dir", type=str, required=True, help="Directory containing rollout JSON files or path to a single JSON")
    args = parser.parse_args()
    
    path = Path(args.json_dir)
    if not path.exists():
        print(f"Path does not exist: {path}")
        return
        
    json_files = []
    if path.is_file() and path.suffix == '.json':
        json_files.append(path)
    elif path.is_dir():
        json_files = list(path.rglob("*.json"))
        
    if not json_files:
        print(f"No JSON files found in {path}")
        return
        
    print(f"Found {len(json_files)} JSON files. Initializing LeanEnv...")
    scheduler = Lean4ServerScheduler()
    lean = LeanEnv(scheduler)
    reward_calc = RewardCalculator()
    assigner = DependencyRewardAssigner(lean, reward_calc)
    
    print("Starting recalculation...")
    for jf in tqdm(json_files):
        process_file(jf, lean, reward_calc, assigner)
        
    print("Recalculation complete.")

if __name__ == "__main__":
    main()
