import os
import json
import glob

def calculate_split_stats(directory, name):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    json_files.sort()
    
    total_problems = len(json_files)
    if total_problems == 0:
        print(f"No files found in {directory}")
        return
        
    total_actions = 0
    total_lean_calls = 0
    total_patch_calls = 0
    
    total_patch_attempted = 0
    total_patch_scored = 0
    
    total_input_chars = 0
    total_output_chars = 0
    
    for path in json_files:
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except Exception:
                continue
                
        # 1. Budget and calls
        meta = data.get("search_metadata", {})
        budget = meta.get("budget", {})
        
        # Actions is the number of expanded action nodes (used_total)
        actions = budget.get("used_total", 0)
        total_actions += actions
        
        # Lean verify calls
        l_calls = budget.get("lean_verify_calls", 0)
        p_calls = budget.get("patch_verify_calls", 0)
        total_lean_calls += l_calls
        total_patch_calls += p_calls
        
        # 2. Patching pipeline
        pipe = meta.get("skeleton_pipeline", {})
        total_patch_attempted += pipe.get("patch_attempted", 0)
        total_patch_scored += pipe.get("patch_scored", 0)
        
        # 3. Token estimation from nodes
        nodes = data.get("nodes", [])
        for n in nodes:
            if n.get("type") == "AND":
                # Input: Prompt text
                prompt = n.get("prompt", "")
                if isinstance(prompt, str):
                    total_input_chars += len(prompt)
                
                # Output: Generated/Extracted code
                code = n.get("extracted_lean_code", "")
                if isinstance(code, str):
                    total_output_chars += len(code)
                
                patched = n.get("patched_code", "")
                if isinstance(patched, str):
                    total_output_chars += len(patched)
                elif isinstance(patched, dict):
                    for val in patched.values():
                        if isinstance(val, str):
                            total_output_chars += len(val)
                            
    # Averages
    avg_actions = total_actions / total_problems
    # Let's check both ways: verify calls only, or including patch calls
    avg_lean_calls_only = total_lean_calls / total_problems
    avg_total_lean_calls = (total_lean_calls + total_patch_calls) / total_problems
    
    # Patching rate: patch_scored / patch_attempted
    patching_success_rate = (total_patch_scored / total_patch_attempted * 100) if total_patch_attempted > 0 else 0
    
    # Token estimation (1 token ~ 4 chars in source code/prompts)
    est_input_tokens = total_input_chars / 4.0
    est_output_tokens = total_output_chars / 4.0
    
    print(f"\n=== Statistics for {name} ({total_problems} problems) ===")
    print(f"Average Actions per problem: {avg_actions:.1f}")
    print(f"Average Lean Verify Calls per problem: {avg_lean_calls_only:.1f}")
    print(f"Average Total Lean Calls (Verify + Patch) per problem: {avg_total_lean_calls:.1f}")
    print(f"Skeleton Patching: Scored {total_patch_scored} / Attempted {total_patch_attempted} ({patching_success_rate:.1f}%)")
    print(f"Total Input Characters: {total_input_chars:,} -> Est. Input Tokens: {est_input_tokens/1_000_000:.2f}M")
    print(f"Total Output Characters: {total_output_chars:,} -> Est. Output Tokens: {est_output_tokens/1_000_000:.2f}M")
    
if __name__ == "__main__":
    calculate_split_stats("/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-valid", "miniF2F-valid")
    calculate_split_stats("/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-test", "miniF2F-test")
