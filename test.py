
import json
import subprocess
import os

def test_hybrid_approach():
    # Context has ↑w, but Goal has explicit (w : ℝ)
    code = """import Mathlib

open Nat Real

theorem my_theorem (x y z w : ℕ) 
  (ht : 1 < x ∧ 1 < y ∧ 1 < z) 
  (hw : 0 < w) 
  (h0 : logb ↑x ↑w = 24) 
  (h1 : logb ↑y ↑w = 40) 
  (h2 : logb (↑x * ↑y * ↑z) ↑w = 12) 
  (h_base_x : logb ↑w ↑x = 1 / 24) 
  (hy_pos : 0 < ↑y) 
  (hy_ne_one : ↑y ≠ 1) 
  (hw_pos : 0 < ↑w) 
  : (w : ℝ) ≠ 1 := by
  sorry
"""
    
    proc = subprocess.Popen(
        ["lake", "exe", "repl"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd="/workspace/npthai/BetaZero/repl",
        bufsize=1,
        preexec_fn=os.setsid
    )
    
    # Warmup
    warmup_cmd = json.dumps({"cmd": "import Mathlib\nset_option linter.unusedVariables false"})
    proc.stdin.write(warmup_cmd + "\n\n")
    proc.stdin.flush()
    
    # Read warmup
    buffer = ""
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        buffer += line
        try:
            json.loads(buffer)
            break
        except json.JSONDecodeError:
            continue
            
    # Send standalone theorem
    payload = {"cmd": code}
    proc.stdin.write(json.dumps(payload) + "\n\n")
    proc.stdin.flush()
    
    # Read result
    buffer = ""
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        buffer += line
        try:
            res = json.loads(buffer)
            os.killpg(os.getpgid(proc.pid), 9)
            
            sorries = res.get("sorries", [])
            if sorries:
                print("=== GOAL AND CONTEXT REPORTED BY LEAN ===")
                print(sorries[0]["goal"])
            return
        except json.JSONDecodeError:
            continue

if __name__ == "__main__":
    test_hybrid_approach()
