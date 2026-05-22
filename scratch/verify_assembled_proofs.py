import os
import glob
import time
from gammazero.env.lean_verifier import Lean4ServerScheduler

def verify_all():
    log_dir = "/workspace/npthai/BetaZero/logs_miniF2F_Test_APOLLO"
    problems = []
    
    # List all subdirectories representing problems
    for entry in os.listdir(log_dir):
        full_path = os.path.join(log_dir, entry)
        if os.path.isdir(full_path):
            problems.append(entry)
            
    print(f"Total problems found in log directory: {len(problems)}")
    
    # We will verify all problems where the assembled_main_theorem.lean exists and does not contain 'sorry'
    candidate_problems = []
    for prob in sorted(problems):
        prob_path = os.path.join(log_dir, prob)
        assembled_file = os.path.join(prob_path, "assembled_main_theorem.lean")
        
        if os.path.isfile(assembled_file):
            with open(assembled_file, "r") as f:
                content = f.read()
            if "sorry" not in content:
                # Find attempt number
                rec_files = glob.glob(os.path.join(prob_path, "**/rec_*.jsonl"), recursive=True)
                attempts = []
                for rf in rec_files:
                    filename = os.path.basename(rf)
                    num = int(filename.split("_")[1].split(".")[0])
                    attempts.append(num)
                max_attempt = max(attempts) if attempts else 1
                
                candidate_problems.append({
                    "name": prob,
                    "file_path": assembled_file,
                    "code": content,
                    "max_attempt": max_attempt
                })
                
    print(f"Found {len(candidate_problems)} candidate solved proofs to verify.")
    
    if not candidate_problems:
        print("No candidates to verify.")
        return
        
    # Start the Lean REPL scheduler in BetaZero
    print("\nBooting Lean REPL Scheduler (4 persistent workers)...")
    scheduler = Lean4ServerScheduler(max_concurrent_requests=4, timeout=60, name="verifier_batch")
    
    try:
        verified_baseline = []
        verified_apollo = []
        failed_verification = []
        
        tasks = [{"code": cand["code"]} for cand in candidate_problems]
        print(f"\nSubmitting {len(tasks)} proofs to verifier queue in parallel batch...")
        
        start_time = time.time()
        req_ids = scheduler.submit_all_request(tasks)
        
        print("\nVerifying in parallel batch (real-time progress)...")
        for idx, (cand, req_id) in enumerate(zip(candidate_problems, req_ids), start=1):
            name = cand["name"]
            category = "Baseline" if cand["max_attempt"] == 1 else "APOLLO"
            
            # Wait for this specific future to resolve
            res = scheduler.futures.pop(req_id).result()
            is_complete = res.get("complete", False)
            
            if is_complete:
                if cand["max_attempt"] > 1:
                    verified_apollo.append(cand)
                else:
                    verified_baseline.append(cand)
                print(f"[{idx}/{len(candidate_problems)}] {name} ({category}) -> SUCCESS")
            else:
                errors = res.get("errors", [])
                err_msg = errors[0].get("data", "Has sorry or warning") if errors else "Has sorry or warning"
                failed_verification.append({
                    "name": name,
                    "error": err_msg,
                    "max_attempt": cand["max_attempt"]
                })
                print(f"[{idx}/{len(candidate_problems)}] {name} ({category}) -> FAILED: {err_msg[:100]}...")
                
        elapsed = time.time() - start_time
        print(f"\nVerification batch finished in {elapsed:.2f} seconds.")
                
        print(f"\n--- VERIFICATION RESULTS SUMMARY ---")
        print(f"Total Candidate Solved Proofs: {len(candidate_problems)}")
        print(f"Successfully Verified Solved Proofs: {len(verified_baseline) + len(verified_apollo)} / {len(candidate_problems)}")
        print(f"  - Verified Baseline (Attempt 1): {len(verified_baseline)} (Success rate over total 244: {len(verified_baseline)/244*100:.2f}%)")
        print(f"  - Verified APOLLO (Attempt > 1): {len(verified_apollo)} (Success rate over total 244: {len(verified_apollo)/244*100:.2f}%)")
        print(f"Failed Verification: {len(failed_verification)}")
        
        if failed_verification:
            print("\n--- Failed Verification Details ---")
            for f in failed_verification:
                cat = "Baseline" if f["max_attempt"] == 1 else "APOLLO"
                print(f"  - {f['name']} ({cat}): {f['error'][:150]}...")
                
    finally:
        scheduler.close()

if __name__ == "__main__":
    verify_all()
