import os
import json
import glob

def analyze_split(directory):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    total_unused_solved = 0
    cat_sibling_failure = 0 # Parent action failed because at least one sibling was not solved
    cat_path_bypass = 0      # Parent action was solved, but parent state chose another solved action
    cat_ancestor_unused = 0  # Parent state was solved, but parent state itself was unused (ancestor failure/bypass)
    
    for path in json_files:
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        
        root_id = data.get("root_id", "state_0")
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        node_status = {}
        node_type = {}
        for n in nodes:
            nid = n["id"]
            node_status[nid] = n["status"]
            node_type[nid] = n["type"]

        expanded_to = {}
        subgoals = {}
        parent_action = {}
        
        for edge in edges:
            src = edge["source"]
            tgt = edge["target"]
            rel = edge["relation"]
            if rel == "expanded_to":
                expanded_to.setdefault(src, []).append(tgt)
            elif rel == "subgoal":
                subgoals.setdefault(src, []).append(tgt)
                parent_action[tgt] = src

        parent_state = {}
        for p_state, acts in expanded_to.items():
            for act in acts:
                parent_state[act] = p_state

        root_solved = (node_status.get(root_id) == "SOLVED")
        used_states = set()
        used_actions = set()
        
        if root_solved:
            queue = [root_id]
            used_states.add(root_id)
            while queue:
                curr_state = queue.pop(0)
                child_actions = expanded_to.get(curr_state, [])
                solved_actions = [act for act in child_actions if node_status.get(act) == "SOLVED"]
                if solved_actions:
                    chosen_action = solved_actions[0]
                    used_actions.add(chosen_action)
                    for child_state in subgoals.get(chosen_action, []):
                        if child_state not in used_states:
                            used_states.add(child_state)
                            queue.append(child_state)
        else:
            queue = [root_id]
            used_states.add(root_id)
            while queue:
                curr_state = queue.pop(0)
                child_actions = expanded_to.get(curr_state, [])
                active_actions = [act for act in child_actions if act in subgoals]
                if active_actions:
                    active_actions.sort(key=lambda act: 2 if node_status.get(act) == "SOLVED" else (1 if node_status.get(act) == "OPEN" else 0), reverse=True)
                    chosen_action = active_actions[0]
                    used_actions.add(chosen_action)
                    for child_state in subgoals.get(chosen_action, []):
                        if child_state not in used_states:
                            used_states.add(child_state)
                            queue.append(child_state)

        for nid, ntype in node_type.items():
            if ntype != "OR" or nid == root_id:
                continue
            
            status = node_status.get(nid)
            is_used = (nid in used_states)
            is_solved = (status == "SOLVED")
            
            if is_solved and not is_used:
                total_unused_solved += 1
                p_act = parent_action.get(nid)
                if not p_act:
                    continue
                
                # Check siblings
                siblings = subgoals.get(p_act, [])
                any_sibling_failed = any(node_status.get(sib) != "SOLVED" for sib in siblings)
                
                p_state = parent_state.get(p_act)
                p_state_solved = (node_status.get(p_state) == "SOLVED")
                
                if any_sibling_failed:
                    cat_sibling_failure += 1
                elif not p_state_solved:
                    # All siblings solved, so parent action is solved. 
                    # If parent state failed, then parent state itself must be unused, meaning it's Category C.
                    cat_ancestor_unused += 1
                else:
                    # All siblings solved, parent state is solved.
                    # If this action was not used by parent state, was another action used?
                    p_state_chosen_act = [act for act in used_actions if parent_state.get(act) == p_state]
                    if p_state_chosen_act and p_state_chosen_act[0] != p_act:
                        cat_path_bypass += 1
                    else:
                        cat_ancestor_unused += 1

    return {
        "total": total_unused_solved,
        "sibling_failure": cat_sibling_failure,
        "path_bypass": cat_path_bypass,
        "ancestor_unused": cat_ancestor_unused
    }

def main():
    valid_dir = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-valid"
    test_dir = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-test"
    
    valid_stats = analyze_split(valid_dir)
    test_stats = analyze_split(test_dir)
    
    print("=== miniF2F-valid ===")
    print(f"Total Solved-but-Unused: {valid_stats['total']}")
    print(f"  - Category A (Sibling Failure): {valid_stats['sibling_failure']} ({valid_stats['sibling_failure']/max(1, valid_stats['total'])*100:.1f}%)")
    print(f"  - Category B (Alternative Path Bypass): {valid_stats['path_bypass']} ({valid_stats['path_bypass']/max(1, valid_stats['total'])*100:.1f}%)")
    print(f"  - Category C (Ancestor Unused/Failed): {valid_stats['ancestor_unused']} ({valid_stats['ancestor_unused']/max(1, valid_stats['total'])*100:.1f}%)")

    print("\n=== miniF2F-test ===")
    print(f"Total Solved-but-Unused: {test_stats['total']}")
    print(f"  - Category A (Sibling Failure): {test_stats['sibling_failure']} ({test_stats['sibling_failure']/max(1, test_stats['total'])*100:.1f}%)")
    print(f"  - Category B (Alternative Path Bypass): {test_stats['path_bypass']} ({test_stats['path_bypass']/max(1, test_stats['total'])*100:.1f}%)")
    print(f"  - Category C (Ancestor Unused/Failed): {test_stats['ancestor_unused']} ({test_stats['ancestor_unused']/max(1, test_stats['total'])*100:.1f}%)")

if __name__ == "__main__":
    main()
