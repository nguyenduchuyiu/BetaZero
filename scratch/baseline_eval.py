import os
import sys
import re
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Ensure BetaZero is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gammazero.env import Lean4ServerScheduler

def extract_lean4_code(text: str) -> str:
    """Extract code wrapped in ```lean4 ... ```, ```lean ... ```, or ``` ... ```."""
    match = re.search(r"```lean4\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```lean\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def evaluate_single_file(lean_file: str, scheduler: Lean4ServerScheduler, client: genai.Client, model: str) -> tuple[bool, int, int]:
    """Evaluates a single Lean 4 file by sampling 32 completions and verifying them."""
    problem_name = os.path.splitext(os.path.basename(lean_file))[0]
    log_dir = "outputs/baseline_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, f"{problem_name}.log")
    json_file_path = os.path.join(log_dir, f"{problem_name}.json")
    
    log_lines = []
    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    log("\n" + "="*80)
    log(f"Evaluating Problem: {lean_file}")
    log("="*80)
    
    try:
        with open(lean_file, "r") as f:
            theorem_content = f.read()
    except Exception as e:
        log(f"❌ Failed to read {lean_file}: {e}")
        return False, 0, 0

    # 1. Define Prompt
    system_instruction = "You are a Lean 4 theorem prover."
    user_prompt = (
        "Given the theorem below, first write a concise proof sketch.\n"
        "Then output the completed Lean 4 code.\n\n"
        "Requirements:\n"
        "- Do not change the theorem statement.\n"
        "- The final Lean code must be inside a ```lean code block.\n"
        "- The final Lean code must compile in Lean 4.\n"
        "- You may use Mathlib and the imported modules.\n"
        "- Do not use 'sorry' or 'admit' in the final proof.\n\n"
        f"{theorem_content}"
    )
    
    log(f"Sampling 32 completions from {model} with temperature=1.0...")
    
    # 2. Sample 32 times in parallel
    samples = []
    def call_gemini(index):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=1.0,
                    max_output_tokens=8192,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            text = response.text or ""
            return index, text
        except Exception as e:
            return index, f"Error calling Gemini: {e}"

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(call_gemini, i) for i in range(32)]
        for fut in as_completed(futures):
            idx, text = fut.result()
            samples.append((idx, text))
            print(f"  [Gemini] Received sample {len(samples)}/32 (elapsed: {time.time() - start_time:.1f}s)", flush=True)
            
    # Sort samples by index
    samples.sort(key=lambda x: x[0])
    
    # Extract Lean 4 code from each sample
    extracted_proofs = []
    for idx, text in samples:
        proof_code = extract_lean4_code(text)
        extracted_proofs.append(proof_code)
        
    # 3. Verify the proofs using the scheduler
    log(f"Submitting 32 proofs to Lean REPL scheduler...")
    start_verify = time.time()
    tasks = [{"code": code} for code in extracted_proofs]
    req_ids = scheduler.submit_all_request(tasks)
    results = scheduler.get_all_request_outputs(req_ids)
    log(f"Verification completed in {time.time() - start_verify:.1f}s\n")
    
    # 4. Analyze and print results
    passed_count = 0
    completed_count = 0
    successful_proof = None
    
    for i, res in enumerate(results):
        is_pass = res.get("pass", False)
        is_complete = res.get("complete", False)
        
        if is_pass:
            passed_count += 1
        if is_complete:
            completed_count += 1
            if successful_proof is None:
                successful_proof = extracted_proofs[i]
                
        log(f"  Sample {i+1:2d}: Pass={str(is_pass):5s} | Complete={str(is_complete):5s} | Time={res.get('verify_time', 0.0):.1f}s")
        if res.get("system_errors"):
            log(f"     [System Error]: {res.get('system_errors')}")
        elif res.get("errors"):
            err = res.get("errors")[0]
            log(f"     [Error]: Line {err.get('pos', {}).get('line', 'N/A')}: {err.get('data', '').strip()}")
            
    log("\n" + "-"*50)
    log(f"Problem Summary ({problem_name}):")
    log(f"  Passed (Typecheck): {passed_count} / 32 ({passed_count/32*100:.1f}%)")
    log(f"  Complete (Proved):  {completed_count} / 32 ({completed_count/32*100:.1f}%)")
    log(f"  Status: {'🎉 SOLVED' if completed_count > 0 else '❌ FAILED'}")
    log("-"*50)
    
    # 5. Write to Human Readable Log File
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== BASELINE EVALUATION LOG ===\n")
        f.write(f"Problem: {lean_file}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("\n".join(log_lines))
        f.write("\n\n=== RAW COMPLETIONS & EXTRACTED PROOFS ===\n")
        for i, (idx, text) in enumerate(samples):
            f.write(f"\n--- SAMPLE {i+1} ---\n")
            f.write(f"[RAW OUTPUT]:\n{text}\n")
            f.write(f"[EXTRACTED CODE]:\n{extracted_proofs[i]}\n")
            f.write(f"[VERIFICATION RESULT]:\n{str(results[i])}\n")
            f.write("-" * 40 + "\n")
            
    # 6. Write to Structured JSON File
    json_data = {
        "problem": lean_file,
        "model": model,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "summary": {
            "total_samples": 32,
            "passed_count": passed_count,
            "completed_count": completed_count,
            "solved": completed_count > 0
        },
        "samples": []
    }
    
    for i, res in enumerate(results):
        is_pass = res.get("pass", False)
        is_complete = res.get("complete", False)
        status = "SOLVED" if is_complete else "FAILED"
        
        sample_entry = {
            "index": i + 1,
            "status": status,
            "raw_text": samples[i][1],
            "extracted_code": extracted_proofs[i],
            "verification": {
                "pass": is_pass,
                "complete": is_complete,
                "errors": res.get("errors", []),
                "warnings": res.get("warnings", []),
                "system_errors": res.get("system_errors", ""),
                "verify_time": res.get("verify_time", 0.0)
            }
        }
        json_data["samples"].append(sample_entry)
        
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
        
    print(f"📝 Log:  {log_file_path}")
    print(f"📊 JSON: {json_file_path}")
    
    return completed_count > 0, passed_count, completed_count

def main():
    load_dotenv()
    # Add elan path to locate 'lake'
    os.environ["PATH"] = f"/root/.elan/bin:{os.environ.get('PATH', '')}"
    
    # 1. Parse arguments
    target_path = "problems/miniF2F-test/amc12b_2002_p3.lean"
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        
    if not os.path.exists(target_path):
        print(f"Error: Path {target_path} does not exist.")
        sys.exit(1)
        
    # 2. Determine file list
    lean_files = []
    if os.path.isdir(target_path):
        print(f"Scanning directory: {target_path}")
        for root, _, files in os.walk(target_path):
            for file in files:
                if file.endswith(".lean"):
                    lean_files.append(os.path.join(root, file))
        lean_files.sort()
    else:
        lean_files = [target_path]
        
    if not lean_files:
        print(f"No .lean files found in {target_path}")
        sys.exit(0)
        
    print(f"Found {len(lean_files)} Lean file(s) to evaluate.")
    
    # 3. Setup Gemini Client
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Missing GOOGLE_API_KEY or GEMINI_API_KEY in environment/.env")
        sys.exit(1)
        
    model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    client = genai.Client(api_key=api_key)
    
    # 4. Boot Lean Server Scheduler once
    print(f"\nBooting Lean REPL Scheduler (8 parallel workers) for batch evaluation...")
    scheduler = Lean4ServerScheduler(max_concurrent_requests=16, timeout=60, name="baseline_verifier")
    
    # 5. Process each file
    global_start = time.time()
    solved_problems = 0
    file_results = []
    
    try:
        for idx, lean_file in enumerate(lean_files):
            print(f"\n[{idx+1}/{len(lean_files)}] processing file...")
            is_solved, passed_count, completed_count = evaluate_single_file(lean_file, scheduler, client, model)
            if is_solved:
                solved_problems += 1
            file_results.append({
                "file": lean_file,
                "solved": is_solved,
                "passed_samples": passed_count,
                "completed_samples": completed_count
            })
    finally:
        scheduler.close()
        
    # 6. Global Summary
    total_time = time.time() - global_start
    print("\n" + "="*80)
    print("GLOBAL EVALUATION SUMMARY")
    print("="*80)
    print(f"Total problems evaluated: {len(lean_files)}")
    print(f"Problems solved:          {solved_problems} / {len(lean_files)} ({solved_problems/len(lean_files)*100:.1f}%)")
    print(f"Total evaluation time:    {total_time:.1f}s (avg {total_time/len(lean_files):.1f}s per problem)")
    print("-"*80)
    
    print(f"{'Problem File':<60} | {'Status':<10} | {'Solved Samples'}")
    print("-"*80)
    for res in file_results:
        status_str = "🎉 SOLVED" if res["solved"] else "❌ FAILED"
        name_truncated = res["file"][-57:] if len(res["file"]) > 60 else res["file"]
        print(f"{name_truncated:<60} | {status_str:<10} | {res['completed_samples']}/32")
    print("="*80)

if __name__ == "__main__":
    main()
