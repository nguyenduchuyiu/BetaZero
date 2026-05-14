import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm

from betazero.core.nodes import ProofState
from betazero.env.lean_env import LeanEnv
from betazero.search.reward.calculator import RewardCalculator
from betazero.search.reward.reward_assigner import DependencyRewardAssigner
from betazero.search.sorrifier.stitcher import ProofStitcher
from betazero.env.lean_env import Lean4ServerScheduler


def has_real_sorry(code: str) -> bool:
    clean = re.sub(r"/\-(?:.|\n)*?\-/|--.*", "", code or "")
    return bool(re.search(r"\bsorry\b", clean))


def extract_sorry_var_order(code: str) -> list[str]:
    """Return skeleton variable names in the same order as their `sorry`s."""
    vars_in_order = []
    seen = set()
    stack = []

    for line in code.splitlines():
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)

        while stack and indent <= stack[-1][0]:
            stack.pop()

        match = re.match(r"(?:have|let)\s+([a-zA-Z0-9_]+)\s*[:=]", stripped)
        if match:
            stack.append((indent, match.group(1)))

        if re.search(r"\bsorry\b", stripped) and stack:
            var_name = stack[-1][1]
            if var_name not in seen:
                vars_in_order.append(var_name)
                seen.add(var_name)

    return vars_in_order


def expand_verifiable_garbage(
    stitched_code: str,
    candidates: set[str],
    garbage_vars: list[str],
    verify_full_code,
    assigner: DependencyRewardAssigner,
) -> tuple[list[str], str]:
    """
    Greedily prune extra skeleton variables if Lean confirms the proof remains
    complete. Do not use this to hide still-unresolved child `sorry` blocks.
    """
    ordered_garbage = list(dict.fromkeys(garbage_vars))
    unresolved_sorry_vars = assigner._extract_sorry_vars(stitched_code)
    cleaned_code = (
        ProofStitcher.prune_garbage(stitched_code, ordered_garbage)
        if ordered_garbage
        else stitched_code
    )

    trial_candidates = candidates - set(ordered_garbage) - unresolved_sorry_vars
    for var in sorted(trial_candidates):
        trial_code = ProofStitcher.prune_garbage(cleaned_code, [var])
        if has_real_sorry(trial_code):
            continue

        trial_vr = verify_full_code(trial_code)
        if trial_vr.get("complete"):
            ordered_garbage.append(var)
            cleaned_code = trial_code

    return ordered_garbage, cleaned_code


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

    updated_count = 0

    def build_full_code(parent_state_data, action_node, proof_code: str) -> str:
        prompt = action_node.get('prompt', '')
        header = "open BigOperators Nat Real Topology\n"
        full_match = None
        match = re.search(r"\[PROBLEM\]\n```lean4\n(.*?)\ntheorem", prompt, re.DOTALL)
        if match:
            header = match.group(1).strip() + "\n"
            full_match = re.search(r"\[PROBLEM\]\n```lean4\n(.*?:= by\n)  sorry\n```", prompt, re.DOTALL)
        if full_match:
            return full_match.group(1) + proof_code

        from betazero.utils.lean_cmd import build_theorem
        ps = ProofState(
            goal=parent_state_data['content']['goal'],
            context=parent_state_data['content'].get('context', ''),
            header=header,
        )
        return build_theorem(ps, proof_code)

    state_eval_cache = {}
    action_eval_cache = {}
    action_live_children = {}
    visiting_states = set()
    visiting_actions = set()

    def set_node_status(node_id: str, status: str):
        nonlocal updated_count
        old = nodes[node_id].get('status')
        if old != status:
            nodes[node_id]['status'] = status
            updated_count += 1

    def set_metric(node_id: str, key: str, value):
        nonlocal updated_count
        metrics = nodes[node_id].setdefault('metrics', {})
        old = metrics.get(key)
        if old != value:
            metrics[key] = value
            updated_count += 1

    def evaluate_action(a_id):
        if a_id in action_eval_cache:
            return action_eval_cache[a_id]
        if a_id in visiting_actions:
            return False, "  sorry"

        visiting_actions.add(a_id)
        a_node = nodes[a_id]
        if a_node['action_type'] == "tactic":
            solved = a_node.get('status') == "SOLVED"
            proof = a_node.get('extracted_lean_code') or "  sorry"
            action_eval_cache[a_id] = (solved, proof)
            visiting_actions.remove(a_id)
            return solved, proof

        extracted_code = a_node.get('extracted_lean_code', '')
        children_ids = action_to_children.get(a_id, [])
        parent_id = action_to_parent.get(a_id)

        if not extracted_code or not children_ids or not parent_id:
            set_node_status(a_id, "FAILED")
            set_metric(a_id, 'r_dep', 0.0)
            action_eval_cache[a_id] = (False, "  sorry")
            visiting_actions.remove(a_id)
            return False, "  sorry"

        child_proofs = []
        for c_id in children_ids:
            child_solved, child_proof = evaluate_state(c_id)
            child_proofs.append(child_proof if child_solved else "  sorry")

        stitched_code = ProofStitcher.stitch(extracted_code, child_proofs)
        parent_state_data = nodes[parent_id]
        full_code = build_full_code(parent_state_data, a_node, stitched_code)

        allowed_vars = assigner._extract_sorry_vars(extracted_code)
        child_vars = extract_sorry_var_order(extracted_code)
        dep_analysis = lean.analyze_dependencies(full_code, allowed_vars=allowed_vars)

        r_dep_score = 0.0

        base_malignant = dep_analysis.get("malignant", [])
        base_benign = dep_analysis.get("benign", [])
        garbage_vars = base_malignant + base_benign
        garbage_vars, cleaned_code = expand_verifiable_garbage(
            stitched_code,
            allowed_vars,
            garbage_vars,
            lambda proof_code: lean.verify(build_full_code(parent_state_data, a_node, proof_code)),
            assigner,
        )

        solved = False
        proof = cleaned_code
        if has_real_sorry(cleaned_code):
            r_dep_score = 0.0
        else:
            cleaned_full_code = build_full_code(parent_state_data, a_node, cleaned_code)
            cleaned_vr = lean.verify(cleaned_full_code)
            if cleaned_vr.get("complete"):
                solved = True
                cleaned_dep_analysis = lean.analyze_dependencies(cleaned_full_code, allowed_vars=allowed_vars)
                if len(cleaned_dep_analysis.get("core_failed", [])) == 0:
                    base_garbage = set(base_malignant) | set(base_benign)
                    extra_benign = sorted(set(garbage_vars) - base_garbage)
                    r_dep_score = reward_calc.r_dep({
                        "core": cleaned_dep_analysis.get("core_solved", []),
                        "benign": cleaned_dep_analysis.get("benign", []) + base_benign + extra_benign,
                        "malignant": cleaned_dep_analysis.get("malignant", []) + base_malignant,
                    })
            else:
                r_dep_score = 0.0

        garbage_set = set(garbage_vars)
        if len(child_vars) == len(children_ids):
            action_live_children[a_id] = [
                c_id for var_name, c_id in zip(child_vars, children_ids)
                if var_name not in garbage_set
            ]
        else:
            action_live_children[a_id] = children_ids

        set_metric(a_id, 'r_dep', r_dep_score)
        set_node_status(a_id, "SOLVED" if solved else "OPEN")
        action_eval_cache[a_id] = (solved, proof if solved else "  sorry")
        visiting_actions.remove(a_id)
        return action_eval_cache[a_id]

    def evaluate_state(s_id):
        nonlocal updated_count

        if s_id in state_eval_cache:
            return state_eval_cache[s_id]
        if s_id in visiting_states:
            return False, "  sorry"

        visiting_states.add(s_id)
        s_node = nodes[s_id]
        candidates = []
        for a_id in state_to_actions.get(s_id, []):
            action_solved, action_proof = evaluate_action(a_id)
            if action_solved:
                candidates.append((a_id, action_proof))

        if candidates:
            clean_candidates = [
                (a_id, proof) for a_id, proof in candidates
                if not has_real_sorry(proof)
            ]
            _, proof = (clean_candidates or candidates)[0]
            set_node_status(s_id, "SOLVED")
            if s_node.get('content', {}).get('proof_body') != proof:
                s_node.setdefault('content', {})['proof_body'] = proof
                updated_count += 1
            result = (True, proof)
        else:
            set_node_status(s_id, "OPEN")
            if s_node.get('content', {}).get('proof_body') != "  sorry":
                s_node.setdefault('content', {})['proof_body'] = "  sorry"
                updated_count += 1
            result = (False, "  sorry")

        state_eval_cache[s_id] = result
        visiting_states.remove(s_id)
        return result

    for s_id, s_node in list(nodes.items()):
        if s_node.get('type') == "OR":
            evaluate_state(s_id)

    # Propagate rewards (Backup) to update Q_value and V_value
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
            c_ids = action_live_children.get(a_id, action_to_children.get(a_id, []))
            if not c_ids:
                future = 0.0
            else:
                future = gamma * min(get_v(c) for c in c_ids)
            res = r_env + solved * (r_dep + future)
            
        visiting_q.remove(a_id)
        q_cache[a_id] = res
        return res

    # Recompute backup for the whole logged graph. Some rollout nodes may no
    # longer be reachable from root after statuses are refreshed, but their
    # stored metrics should still be internally consistent.
    root_id = data.get('root_id')
    if root_id:
        get_v(root_id)

        for node_id, node in nodes.items():
            if node.get('type') == "AND":
                get_q(node_id)
        for node_id, node in nodes.items():
            if node.get('type') == "OR":
                get_v(node_id)

        # Cập nhật lại metrics trong nodes
        for a_id, q_val in q_cache.items():
            nodes[a_id].setdefault('metrics', {})['Q_value'] = q_val
        for s_id, v_val in v_cache.items():
            nodes[s_id].setdefault('metrics', {})['V_value'] = v_val

        for s_id, s_node in nodes.items():
            if s_node.get('type') != "OR" or s_node.get('status') != "SOLVED":
                continue

            solved_actions = [
                a_id for a_id in state_to_actions.get(s_id, [])
                if nodes[a_id].get('status') == "SOLVED"
            ]
            clean_actions = [
                a_id for a_id in solved_actions
                if not has_real_sorry(action_eval_cache.get(a_id, (False, "  sorry"))[1])
            ]
            candidates = clean_actions or solved_actions
            if not candidates:
                continue

            best_action = max(candidates, key=lambda a_id: q_cache.get(a_id, nodes[a_id].get('metrics', {}).get('Q_value', 0.0)))
            best_proof = action_eval_cache.get(best_action, (False, None))[1]
            if best_proof and s_node.get('content', {}).get('proof_body') != best_proof:
                s_node.setdefault('content', {})['proof_body'] = best_proof
                updated_count += 1
            
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
