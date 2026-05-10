import json
import sys
import os
from tqdm import tqdm
from copy import deepcopy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from betazero.search.sorrifier.stitcher import ProofStitcher
from betazero.env.expr_parser import get_lean_expr_tree
from betazero.search.sorrifier.dependency_analyzer import SHARED_EXPR_ANALYZER
from betazero.search.reward.calculator import RewardCalculator
from betazero.utils.lean_cmd import build_theorem
from betazero.core.nodes import ProofState

class StandardRewardReconstructor:
    def __init__(self, input_dir, output_dir, W_solve=1.0, gamma=1.0):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.W_solve = W_solve
        self.gamma = gamma
        os.makedirs(output_dir, exist_ok=True)
        self.calculator = RewardCalculator()

    def process_file(self, filename):
        input_path = os.path.join(self.input_dir, filename)
        output_path = os.path.join(self.output_dir, filename)
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        nodes = {n['id']: n for n in data['nodes']}
        edges = data.get('edges', [])
        
        # Build adjacency
        children = {} 
        parent_state_id = {} 
        state_actions = {} 
        
        for edge in edges:
            src, tgt, rel = edge['source'], edge['target'], edge['relation']
            if rel == 'expanded_to':
                if src not in state_actions: state_actions[src] = []
                state_actions[src].append(tgt)
                parent_state_id[tgt] = src
            elif rel == 'subgoal':
                if src not in children: children[src] = []
                children[src].append(tgt)

        # 1. Reset & Calculate r_env
        for node in nodes.values():
            if 'metrics' not in node or node['metrics'] is None:
                node['metrics'] = {}
            
            if node['type'] == 'AND':
                lean_code = node.get('extracted_lean_code', '')
                p_id = parent_state_id.get(node['id'])
                if lean_code and p_id:
                    p_node = nodes[p_id]
                    st = ProofState(
                        context=p_node['content']['context'],
                        goal=p_node['content']['goal'],
                        header="import Mathlib\nopen Real"
                    )
                    full_code = build_theorem(st, lean_code)
                    node['metrics']['r_env'] = self.calculator.r_env(full_code, full_code, {})
                else:
                    node['metrics']['r_env'] = 0.0
                node['metrics']['r_dep'] = 0.0
            else:
                node['metrics']['V_value'] = 0.0

        # 2. Propagate Status (Ensure SOLVED is correct)
        for _ in range(10): 
            changed = False
            for node_id, node in nodes.items():
                if node['status'] == 'SOLVED': continue
                
                if node['type'] == 'AND':
                    if node.get('action_type') == 'skeleton':
                        subgoals = children.get(node_id, [])
                        if subgoals and all(nodes[sid]['status'] == 'SOLVED' for sid in subgoals):
                            node['status'] = 'SOLVED'
                            changed = True
                elif node['type'] == 'OR':
                    actions = state_actions.get(node_id, [])
                    if any(nodes[aid]['status'] == 'SOLVED' for aid in actions):
                        node['status'] = 'SOLVED'
                        changed = True
            if not changed: break

        # 3. Stitch & Calculate r_dep for ALL skeletons (Not just solved ones!)
        all_skeletons = [n for n in nodes.values() if n['type'] == 'AND' and n.get('action_type') == 'skeleton']

        for node in all_skeletons:
            node_id = node['id']
            subgoal_states = children.get(node_id, [])
            child_proofs = []
            
            # Get child proof codes (or None if not solved)
            for sg_id in subgoal_states:
                solved_action = None
                for act_id in state_actions.get(sg_id, []):
                    if nodes[act_id]['status'] == 'SOLVED':
                        solved_action = nodes[act_id]
                        break
                
                if solved_action:
                    if solved_action.get('action_type') == 'skeleton':
                        child_proofs.append(solved_action.get('content', ""))
                    else:
                        child_proofs.append(solved_action.get('extracted_lean_code', ""))
                else:
                    child_proofs.append(None) # None will be replaced by 'sorry' during stitching
            
            # Stitch even if there are Nones
            extracted = node.get('extracted_lean_code', '')
            stitched = ProofStitcher.stitch(extracted, child_proofs)
            p_id = parent_state_id.get(node_id)
            if not p_id: continue
            p_node = nodes[p_id]
            
            st = ProofState(
                context=p_node['content']['context'],
                goal=p_node['content']['goal'],
                header="import Mathlib\nopen Real" 
            )
            full_code = build_theorem(st, stitched)
            
            # Analyze through Kernel Expr Tree
            expr_results = get_lean_expr_tree(full_code)
            
            if expr_results:
                root_expr = expr_results[-1].get("expr_value_tree")
                classification = SHARED_EXPR_ANALYZER.classify_skeleton_subgoals(root_expr)
                
                if len(classification.get("core_failed", [])) > 0:
                    node['metrics']['r_dep'] = 0.0
                else:
                    mapped_analysis = {
                        "core": classification.get("core_solved", []),
                        "benign": classification.get("benign", []),
                        "malignant": classification.get("malignant", [])
                    }
                    node['metrics']['r_dep'] = self.calculator.r_dep(mapped_analysis)
                    if extracted:
                        node['status'] = 'SOLVED'

        # 3.5 Re-propagate Status since some skeletons might have been newly marked as SOLVED
        for _ in range(10):
            changed = False
            for node_id, node in nodes.items():
                if node['status'] == 'SOLVED': continue
                if node['type'] == 'OR':
                    actions = state_actions.get(node_id, [])
                    if any(nodes[aid]['status'] == 'SOLVED' for aid in actions):
                        node['status'] = 'SOLVED'
                        changed = True
                elif node['type'] == 'AND' and node.get('action_type') == 'skeleton':
                    subgoals = children.get(node_id, [])
                    if subgoals and all(nodes[sid]['status'] == 'SOLVED' for sid in subgoals):
                        node['status'] = 'SOLVED'
                        changed = True
            if not changed: break

        # 4. Final Reward Propagation (Q and V)
        for _ in range(20): 
            for node_id, node in nodes.items():
                if node['type'] == 'AND':
                    r_e = node['metrics'].get('r_env', 0.0)
                    r_d = node['metrics'].get('r_dep', 0.0)
                    solved = (node['status'] == 'SOLVED')
                    
                    if node.get('action_type') == 'tactic':
                        node['metrics']['Q_value'] = r_e + self.W_solve * float(solved)
                    else:
                        child_v = [nodes[sid]['metrics'].get('V_value', 0.0) for sid in children.get(node_id, [])]
                        future = min(child_v) if child_v else 0.0
                        node['metrics']['Q_value'] = r_e + float(solved) * (r_d + self.gamma * future)
                
                elif node['type'] == 'OR':
                    child_qs = [nodes[aid]['metrics'].get('Q_value', 0.0) for aid in state_actions.get(node_id, [])]
                    node['metrics']['V_value'] = max(child_qs) if child_qs else 0.0

        # Save
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def run(self):
        files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.json')])
        for f in tqdm(files, desc="Re-calculating Standard Rewards"):
            try:
                self.process_file(f)
            except Exception as e:
                print(f"\nError in {f}: {e}")

if __name__ == "__main__":
    recon = StandardRewardReconstructor(
        # "outputs/rollouts/deepseekr1qwen/miniF2F-valid-100",
        # "deepseekr1qwen/miniF2F-valid-100"
        "outputs/rollouts/deepseekv4/miniF2F-valid-100",
        "deepseekv4/miniF2F-valid-100"
    )
    recon.run()
