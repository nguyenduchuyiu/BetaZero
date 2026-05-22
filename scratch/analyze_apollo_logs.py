import os
import glob

def analyze():
    log_dir = "/workspace/npthai/BetaZero/logs_miniF2F_Test_APOLLO"
    problems = []
    
    # List all subdirectories representing problems
    for entry in os.listdir(log_dir):
        full_path = os.path.join(log_dir, entry)
        if os.path.isdir(full_path):
            problems.append(entry)
            
    print(f"Total problems found in log directory: {len(problems)}")
    
    solved_problems = []
    failed_problems = []
    
    solved_baseline = [] # Solved at attempt 1
    solved_apollo = []   # Solved at attempt > 1
    
    for prob in sorted(problems):
        prob_path = os.path.join(log_dir, prob)
        assembled_file = os.path.join(prob_path, "assembled_main_theorem.lean")
        
        is_solved = False
        if os.path.isfile(assembled_file):
            with open(assembled_file, "r") as f:
                content = f.read()
            # A problem is solved if it does not contain 'sorry'
            if "sorry" not in content:
                is_solved = True
                
        if is_solved:
            solved_problems.append(prob)
            
            # Find the highest attempt number
            rec_files = glob.glob(os.path.join(prob_path, "**/rec_*.jsonl"), recursive=True)
            attempts = []
            for rf in rec_files:
                filename = os.path.basename(rf)
                num = int(filename.split("_")[1].split(".")[0])
                attempts.append(num)
                
            max_attempt = max(attempts) if attempts else 1
            
            if max_attempt > 1:
                solved_apollo.append((prob, max_attempt))
            else:
                solved_baseline.append(prob)
        else:
            failed_problems.append(prob)
            
    print(f"\n--- RESULTS SUMMARY ---")
    print(f"Total problems: {len(problems)}")
    print(f"Solved problems: {len(solved_problems)} ({len(solved_problems)/len(problems)*100:.2f}%)")
    print(f"Failed problems: {len(failed_problems)}")
    print(f"  - Solved by Baseline (Attempt 1): {len(solved_baseline)} ({len(solved_baseline)/len(problems)*100:.2f}%)")
    print(f"  - Solved by APOLLO (Attempt > 1): {len(solved_apollo)} ({len(solved_apollo)/len(problems)*100:.2f}%)")
    
    print("\n--- Solved by Baseline (Attempt 1) ---")
    for p in solved_baseline:
        print(f"  - {p}")
        
    print("\n--- Solved by APOLLO (Attempt > 1) ---")
    for p, max_att in solved_apollo:
        print(f"  - {p} (Max Attempt: {max_att})")

if __name__ == "__main__":
    analyze()
