import os
import json
import glob

def find_solved_but_unused_in_dir(directory, split_name):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    results = []

    for path in sorted(json_files):
        filename = os.path.basename(path)
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                continue
        
        root_id = data.get("root_id", "state_0")
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        node_status = {}
        node_type = {}
        node_depth = {}
        node_goal = {}
        node_action_type = {}
        node_content = {}
        node_label = {}
        
        for n in nodes:
            nid = n["id"]
            node_status[nid] = n["status"]
            node_type[nid] = n["type"]
            node_depth[nid] = n.get("depth", 0)
            node_label[nid] = n.get("target_label", "")
            
            content = n.get("content", "")
            if isinstance(content, dict):
                node_goal[nid] = content.get("goal", "")
                node_content[nid] = content.get("proof_body", "")
            else:
                node_goal[nid] = n.get("goal", "")
                node_content[nid] = content
                
            node_action_type[nid] = n.get("action_type", "")

        expanded_to = {}
        subgoals = {}
        parent_action = {} # child_state -> parent_action
        
        for edge in edges:
            src = edge["source"]
            tgt = edge["target"]
            rel = edge["relation"]
            if rel == "expanded_to":
                expanded_to.setdefault(src, []).append(tgt)
            elif rel == "subgoal":
                subgoals.setdefault(src, []).append(tgt)
                parent_action[tgt] = src

        # Find parent state of action
        parent_state = {} # action -> parent_state
        for p_state, acts in expanded_to.items():
            for act in acts:
                parent_state[act] = p_state

        root_solved = (node_status.get(root_id) == "SOLVED")
        
        # Traverse to find used states
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

        # Look for solved but unused subgoals
        for nid, ntype in node_type.items():
            if ntype != "OR" or nid == root_id:
                continue
            
            status = node_status.get(nid)
            is_used = (nid in used_states)
            is_solved = (status == "SOLVED")
            
            if is_solved and not is_used:
                # Find its parent action
                p_act = parent_action.get(nid)
                if p_act:
                    p_state = parent_state.get(p_act)
                    p_state_goal = node_goal.get(p_state, "")
                    p_act_content = node_content.get(p_act, "")
                    
                    results.append({
                        "file": filename,
                        "subgoal_id": nid,
                        "subgoal_label": node_label.get(nid, ""),
                        "subgoal_goal": node_goal.get(nid, ""),
                        "parent_state_id": p_state,
                        "parent_state_label": node_label.get(p_state, ""),
                        "parent_state_goal": p_state_goal,
                        "parent_action_id": p_act,
                        "parent_action_content": p_act_content
                    })

    return results

def main():
    valid_dir = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-valid"
    test_dir = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-test"
    
    print("--- Searching for Solved But Unused Subgoals in miniF2F-valid ---")
    valid_results = find_solved_but_unused_in_dir(valid_dir, "valid")
    print(f"Found {len(valid_results)} solved but unused subgoals.")
    for idx, r in enumerate(valid_results[:10]):
        print(f"\n[{idx+1}] File: {r['file']}")
        print(f"  Unused Subgoal ID: {r['subgoal_id']} (Label: {r['subgoal_label']})")
        print(f"  Unused Subgoal Goal: {r['subgoal_goal']}")
        print(f"  Parent State ID (Mẹ): {r['parent_state_id']} (Label: {r['parent_state_label']})")
        print(f"  Parent State Goal (Goal Mẹ): {r['parent_state_goal']}")
        print(f"  Parent Action Content: {r['parent_action_content'][:200]}...")

    print("\n--- Searching for Solved But Unused Subgoals in miniF2F-test ---")
    test_results = find_solved_but_unused_in_dir(test_dir, "test")
    print(f"Found {len(test_results)} solved but unused subgoals.")
    for idx, r in enumerate(test_results[:10]):
        print(f"\n[{idx+1}] File: {r['file']}")
        print(f"  Unused Subgoal ID: {r['subgoal_id']} (Label: {r['subgoal_label']})")
        print(f"  Unused Subgoal Goal: {r['subgoal_goal']}")
        print(f"  Parent State ID (Mẹ): {r['parent_state_id']} (Label: {r['parent_state_label']})")
        print(f"  Parent State Goal (Goal Mẹ): {r['parent_state_goal']}")
        print(f"  Parent Action Content: {r['parent_action_content'][:200]}...")

if __name__ == "__main__":
    main()
