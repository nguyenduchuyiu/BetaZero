import os

def fix_in_symbol():
    log_dir = "/workspace/npthai/BetaZero/logs_miniF2F_Test_APOLLO"
    modified_files = []
    
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f == "assembled_main_theorem.lean" or f == "main_theorem.lean":
                file_path = os.path.join(root, f)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                
                # We target " in Finset" or " in " specifically
                new_content = content
                if " in Finset" in new_content:
                    new_content = new_content.replace(" in Finset", " ∈ Finset")
                if " in range" in new_content:
                    new_content = new_content.replace(" in range", " ∈ range")
                    
                if new_content != content:
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(new_content)
                    modified_files.append(file_path)
                    print(f"Fixed 'in' -> '∈' in: {file_path}")
                    
    print(f"\nCompleted! Modified {len(modified_files)} files.")

if __name__ == "__main__":
    fix_in_symbol()
