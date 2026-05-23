import os
import json

def merge_all_action_contents():
    json_dir = "/workspace/BetaZero/outputs/rollouts/gemini3flash/miniF2F-test"
    output_path = "/workspace/BetaZero/miniF2F_test_solved_contents.txt"
    
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
    merged_contents = []
    
    solved_count = 0
    total_actions_merged = 0
    
    for f in json_files:
        json_path = os.path.join(json_dir, f)
        with open(json_path, "r", encoding="utf-8") as jf:
            data = json.load(jf)
            
        root_id = data.get("root_id", "state_0")
        root_node = next((n for n in data.get("nodes", []) if n.get("id") == root_id), None)
        
        # Only process successfully solved problems
        if root_node and root_node.get("status") == "SOLVED":
            solved_count += 1
            # Extract content from ALL AND (Action) nodes in the tree
            for node in data.get("nodes", []):
                if node.get("type") == "AND":
                    content = node.get("content", "")
                    if content and isinstance(content, str):
                        merged_contents.append(content)
                        total_actions_merged += 1
                        
    # Concatenate all content fields with exactly a newline
    final_text = "\n".join(merged_contents)
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write(final_text)
        
    print(f"Successfully merged {total_actions_merged} total action content fields from {solved_count} solved problems!")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    merge_all_action_contents()
