import json
import os

def export_lean_problems(json_path, output_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    valid_dir = os.path.join(output_dir, "miniF2F-valid")
    test_dir = os.path.join(output_dir, "miniF2F-test")
    
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    count_valid = 0
    count_test = 0
    
    for item in data:
        name = item.get("name")
        split = item.get("split")
        formal_statement = item.get("formal_statement")
        header = item.get("header", "")
        
        if not name or not formal_statement:
            continue
            
        # Ensure header and formal statement are formatted nicely
        content = header.strip() + "\n\n" + formal_statement.strip() + "\n  sorry\n"
        
        target_dir = valid_dir if split == "valid" else test_dir
        file_path = os.path.join(target_dir, f"{name}.lean")
        
        with open(file_path, 'w', encoding='utf-8') as f_out:
            f_out.write(content)
            
        if split == "valid":
            count_valid += 1
        else:
            count_test += 1

    print(f"Done! Exported {count_valid} valid and {count_test} test theorems to {output_dir}/")

if __name__ == "__main__":
    json_file = "/workspace/npthai/BetaZero/miniF2F_v2s.json"
    output_base = "/workspace/npthai/BetaZero/problems"
    export_lean_problems(json_file, output_base)
