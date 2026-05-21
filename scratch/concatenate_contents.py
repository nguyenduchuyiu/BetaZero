import os
import json
import glob

def concatenate_separated(directory, output_dir):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    print(f"Reading {len(json_files)} JSON files...")

    input_file = os.path.join(output_dir, "all_llm_inputs.txt")
    output_file = os.path.join(output_dir, "all_llm_outputs.txt")

    total_prompts = 0
    total_completions = 0
    
    with open(input_file, "w", encoding="utf-8") as out_in, open(output_file, "w", encoding="utf-8") as out_out:
        for path in sorted(json_files):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
                
                nodes = data.get("nodes", [])
                for node in nodes:
                    if node.get("type") == "AND":
                        prompt = node.get("prompt", "")
                        content = node.get("content", "")
                        
                        out_in.write(prompt)
                        out_out.write(content)
                        
                        total_prompts += len(prompt)
                        total_completions += len(content)

    print(f"\nSuccessfully concatenated inputs and outputs separately!")
    print(f"Inputs saved to: {input_file}")
    print(f"Outputs saved to: {output_file}")
    print(f"Total prompt characters: {total_prompts}")
    print(f"Total completion characters: {total_completions}")
    est_prompt_tokens = total_prompts // 4
    est_comp_tokens = total_completions // 4
    print(f"Estimated Prompt (Input) Tokens: ~{est_prompt_tokens:,}")
    print(f"Estimated Completion (Output) Tokens: ~{est_comp_tokens:,}")

if __name__ == "__main__":
    concatenate_separated(
        "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-test",
        "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-test"
    )
