import os
import json

def investigate(path):
    with open(path, "r") as f:
        data = json.load(f)
        
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    
    node_status = {}
    node_type = {}
    node_label = {}
    node_goal = {}
    
    for n in nodes:
        nid = n["id"]
        node_status[nid] = n["status"]
        node_type[nid] = n["type"]
        node_label[nid] = n.get("target_label", "")
        content = n.get("content", "")
        if isinstance(content, dict):
            node_goal[nid] = content.get("goal", "")
        else:
            node_goal[nid] = n.get("goal", "")

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

    # Traverse to find used states
    root_id = data.get("root_id", "state_0")
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

    print(f"=== Investigating: {os.path.basename(path)} ===")
    for nid, ntype in node_type.items():
        if ntype != "OR" or nid == root_id:
            continue
        
        status = node_status.get(nid)
        is_used = (nid in used_states)
        is_solved = (status == "SOLVED")
        
        if is_solved and not is_used:
            p_act = parent_action.get(nid)
            if p_act:
                p_state = parent_state.get(p_act)
                siblings = subgoals.get(p_act, [])
                
                print(f"\nUnused Solved Subgoal: {nid} (Label: {node_label.get(nid)}, Goal: {node_goal.get(nid)})")
                print(f"Parent State (Mẹ): {p_state} (Label: {node_label.get(p_state)}, Status: {node_status.get(p_state)}, Goal: {node_goal.get(p_state)})")
                print(f"Parent Action (AND Node): {p_act} (Status: {node_status.get(p_act)}, Action Type: {node_status.get(p_act)})")
                print("Siblings:")
                for sib in siblings:
                    sib_status = node_status.get(sib)
                    sib_used = (sib in used_states)
                    print(f"  - Sibling Subgoal {sib} (Label: {node_label.get(sib)}, Status: {sib_status}, Used: {sib_used}, Goal: {node_goal.get(sib)})")

if __name__ == "__main__":
    path = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-valid/algebra_apbmpcneq0_aeq0anbeq0anceq0.json"
    investigate(path)
