import os

def parse_all_python_files_and_write_to_code_md(root: str):
    code_lines = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.py') and file != os.path.basename(__file__):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_lines.append(f"# File: {file_path}\n")
                        code_lines.extend(f.readlines())
                        code_lines.append('\n\n')
                except Exception as e:
                    code_lines.append(f"# Could not read {file_path}: {e}\n\n")
    with open("code.md", "w", encoding="utf-8") as code_md:
        code_md.writelines(code_lines)

if __name__ == "__main__":
    parse_all_python_files_and_write_to_code_md(root="betazero")