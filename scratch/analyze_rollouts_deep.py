import os
import json
import glob

def analyze_deep(directory):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    print(f"Analyzing {len(json_files)} rollout JSON files for deep metrics...")

    total_theorems = len(json_files)
    solved_theorems = 0
    stitching_success_count = 0
    
    total_skeletons_inserted = 0
    solved_direct_only = 0
    solved_with_skeleton = 0
    
    # Skeleton pipeline extraction metrics
    total_requested_skeletons = 0
    total_patch_failed = 0
    
    # Dependency classes
    total_solved_and_used = 0
    total_solved_but_unused = 0
    total_unresolved_unused = 0
    total_unresolved_used = 0

    theorem_details = []

    for path in sorted(json_files):
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        name = os.path.basename(path).replace(".json", "")
        root_id = data.get("root_id", "state_0")
        
        # 1. Pipeline metrics
        pipe = data.get("search_metadata", {}).get("skeleton_pipeline", {})
        total_requested_skeletons += pipe.get("requested", 0)
        total_patch_failed += pipe.get("patch_failed", 0)
        
        # 2. Graph reconstruction
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        node_status = {}
        node_type = {}
        node_depth = {}
        for n in nodes:
            nid = n["id"]
            node_status[nid] = n["status"]
            node_type[nid] = n["type"]
            node_depth[nid] = n.get("depth", 0)
            if n.get("action_type") == "skeleton":
                total_skeletons_inserted += 1

        expanded_to = {}  # state_id -> list of action_id
        subgoals = {}  # action_id -> list of state_id
        for edge in edges:
            src = edge["source"]
            tgt = edge["target"]
            rel = edge["relation"]
            if rel == "expanded_to":
                expanded_to.setdefault(src, []).append(tgt)
            elif rel == "subgoal":
                subgoals.setdefault(src, []).append(tgt)

        root_solved = (node_status.get(root_id) == "SOLVED")
        if root_solved:
            solved_theorems += 1
            stitching_success_count += 1  # If root is solved, final stitching succeeded

        # 3. Solved by direct vs skeleton
        max_solved_depth = 0
        for nid, status in node_status.items():
            if node_type[nid] == "OR" and status == "SOLVED":
                max_solved_depth = max(max_solved_depth, node_depth.get(nid, 0))
        
        if root_solved:
            if max_solved_depth == 0:
                solved_direct_only += 1
            else:
                solved_with_skeleton += 1

        # 4. Dependency Class Classification via Graph Traversal
        used_states = set()
        used_actions = set()
        
        if root_solved:
            # Solved traversal
            queue = [root_id]
            used_states.add(root_id)
            while queue:
                curr_state = queue.pop(0)
                child_actions = expanded_to.get(curr_state, [])
                # Find a solved action child
                solved_actions = [act for act in child_actions if node_status.get(act) == "SOLVED"]
                if solved_actions:
                    # Pick the first one (or highest priority/r_dep if available)
                    chosen_action = solved_actions[0]
                    used_actions.add(chosen_action)
                    for child_state in subgoals.get(chosen_action, []):
                        if child_state not in used_states:
                            used_states.add(child_state)
                            queue.append(child_state)
        else:
            # Unsolved traversal (trace committed/active paths)
            queue = [root_id]
            used_states.add(root_id)
            while queue:
                curr_state = queue.pop(0)
                child_actions = expanded_to.get(curr_state, [])
                # An action is active if it has subgoal child nodes in the graph
                active_actions = [act for act in child_actions if act in subgoals]
                if active_actions:
                    # Pick the best active action (prefer solved, then open)
                    active_actions.sort(key=lambda act: 2 if node_status.get(act) == "SOLVED" else (1 if node_status.get(act) == "OPEN" else 0), reverse=True)
                    chosen_action = active_actions[0]
                    used_actions.add(chosen_action)
                    for child_state in subgoals.get(chosen_action, []):
                        if child_state not in used_states:
                            used_states.add(child_state)
                            queue.append(child_state)

        # Count state nodes (excluding the root)
        file_solved_and_used = 0
        file_solved_but_unused = 0
        file_unresolved_unused = 0
        file_unresolved_used = 0

        for nid, ntype in node_type.items():
            if ntype != "OR" or nid == root_id:
                continue
            
            status = node_status.get(nid)
            is_used = (nid in used_states)
            is_solved = (status == "SOLVED")
            
            if is_used:
                if is_solved:
                    file_solved_and_used += 1
                else:
                    file_unresolved_used += 1
            else:
                if is_solved:
                    file_solved_but_unused += 1
                else:
                    file_unresolved_unused += 1

        total_solved_and_used += file_solved_and_used
        total_solved_but_unused += file_solved_but_unused
        total_unresolved_unused += file_unresolved_unused
        total_unresolved_used += file_unresolved_used

        theorem_details.append({
            "name": name,
            "solved": root_solved,
            "max_solved_depth": max_solved_depth if root_solved else "-",
            "solved_and_used": file_solved_and_used,
            "solved_but_unused": file_solved_but_unused,
            "unresolved_unused": file_unresolved_unused,
            "unresolved_used": file_unresolved_used,
        })

    subgoal_extraction_failure_rate = (total_patch_failed / total_requested_skeletons) * 100 if total_requested_skeletons > 0 else 0
    avg_skeletons = total_skeletons_inserted / total_theorems if total_theorems > 0 else 0
    
    print("\n================== DEEP METRICS ANALYSIS ==================")
    print(f"Total Solved Theorems: {solved_theorems}")
    print(f"Final Stitching Success Count: {stitching_success_count} / {solved_theorems} (100.00% of solved theorems)")
    print(f"Total Inserted Skeletons: {total_skeletons_inserted} (Avg per run: {avg_skeletons:.2f})")
    print(f"Solved by Direct Local Proof only: {solved_direct_only}")
    print(f"Solved with Skeleton Decomposition: {solved_with_skeleton}")
    print(f"Subgoal Extraction Failure Rate: {total_patch_failed} / {total_requested_skeletons} ({subgoal_extraction_failure_rate:.2f}%)")
    print("\n--- Subgoal Dependency Classes ---")
    print(f"Solved-and-Used:    {total_solved_and_used}")
    print(f"Solved-but-Unused:  {total_solved_but_unused}")
    print(f"Unresolved-Unused:  {total_unresolved_unused}")
    print(f"Unresolved-Used:    {total_unresolved_used}")
    print("===========================================================")

    # Output detailed per-theorem dependency table
    print("\n### Subgoal Dependency Classification Table:")
    print("| Theorem | Solved? | Solved-and-Used | Solved-but-Unused | Unresolved-Unused | Unresolved-Used |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for td in theorem_details:
        status = "✅ YES" if td["solved"] else "❌ NO"
        print(f"| {td['name']} | {status} | {td['solved_and_used']} | {td['solved_but_unused']} | {td['unresolved_unused']} | {td['unresolved_used']} |")

if __name__ == "__main__":
    analyze_deep("/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-valid")
