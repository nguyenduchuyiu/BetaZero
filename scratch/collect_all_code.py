import os

def collect_code(root_dir, output_file):
    exclude_dirs = {'.git', '__pycache__', 'outputs', 'data', 'scratch', 'unsloth_compiled_cache', '.gemini'}
    # Common code and config extensions
    include_extensions = {'.py', '.yaml', '.yml', '.sh', '.html', '.css', '.js', '.json', '.txt', '.md'}
    # Specific files to exclude (e.g., large logs or temporary files)
    exclude_files = {'gpu_watcher.log', 'out.txt', 'gem.txt', 'test_ske.txt', 'test_tac.txt'}

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for root, dirs, files in os.walk(root_dir):
            # Remove excluded directories from search
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files:
                    continue
                
                ext = os.path.splitext(file)[1].lower()
                if ext in include_extensions:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, root_dir)
                    
                    try:
                        # Check file size to avoid dumping massive data files that might have text extensions
                        if os.path.getsize(file_path) > 500000: # 500KB limit for safety
                            print(f"Skipping large file: {rel_path}")
                            continue

                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                            content = f_in.read()
                            
                        f_out.write(f"\n{'='*80}\n")
                        f_out.write(f"FILE: {rel_path}\n")
                        f_out.write(f"{'='*80}\n\n")
                        f_out.write(content)
                        f_out.write("\n")
                        print(f"Added: {rel_path}")
                    except Exception as e:
                        print(f"Error reading {rel_path}: {e}")

if __name__ == "__main__":
    root = "/workspace/npthai/BetaZero/betazero"
    output = "/workspace/npthai/BetaZero/betazero_all_code.txt"
    collect_code(root, output)
    print(f"\nDone! All code collected in {output}")
