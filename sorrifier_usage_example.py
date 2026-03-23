from prover.lean.verifier import Lean4ServerScheduler
from utils.auto_sorrifier import AutoSorrifier, AST_DAEMON, REPL_DIR
from utils.syntax_repair import SyntaxCorrector
import os
import tempfile
from datetime import datetime

code = '''
import Mathlib
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat

theorem log_b_clean (a b : ℝ) 
  (h₀ : Real.logb 8 a + Real.logb 4 (b ^ 2) = 5)
  (h₁ : Real.logb 8 b + Real.logb 4 (a ^ 2) = 7)
  : a * b = 512 := by
  -- Step 1: Convert Logarithms to Base 2
  -- Step 2: Simplify the Equations
  -- Step 3: Introduce Variables for Simplification
  -- Step 4: Solve the System of Equations
  -- Step 5: Exponentiate to Find a and b
  -- Step 6: Calculate the Product a * b
  have h1 : (1/3 : ℝ) * Real.logb 2 a + Real.logb 2 b = 5 := by
    -- Step 1 and 2: Convert and simplify the logarithmic expressions
    field_simp [Real.logb, log_mul, mul_add, mul_comm, mul_left_comm] at h1 ⊢
    -- Step 3: Introduce variables for simplification
    ring_nf at h1 ⊢
    -- Step 4: Solve the system of equations using linarith
    linarith
  have h2 : (1/3 : ℝ) * Real.logb 2 b + Real.logb 2 a = 7 := by
    -- Step 1 and 2: Convert and simplify the logarithmic expressions
    field_simp [Real.logb, log_mul, mul_add, mul_comm, mul_left_comm] at h2 ⊢
    -- Step 3: Introduce variables for simplification
    ring_nf at h2 ⊢
    -- Step 4: Solve the system of equations using linarith
    linarith
  -- Step 5 and 6: Calculate the product a * b and verify it equals 512
  have h3 : a * b = 512 := by
    -- Use the properties of logarithms and arithmetic to solve for a and b
    have h4 : a = 64 := by
      -- Solve for a using the simplified equations
      nlinarith
    have h5 : b = 8 := by
      -- Solve for b using the simplified equations
      nlinarith
    -- Calculate the product a * b
    rw [h4, h5]
    norm_num
  -- Final verification that a * b = 512
  exact h3

'''

def find_unused_haves(ast, code):
    haves = []

    # collect hypothesis definitions
    for i, node in enumerate(ast):
        if node["kind"] == "have":
            name_node = ast[i+4]  # ident của h
            name = code[name_node["start_byte"]:name_node["end_byte"]]

            haves.append({
                "name": name,
                "pos": name_node["end_byte"]
            })

    unused = []

    for h in haves:
        name = h["name"]
        pos = h["pos"]

        used = False
        for node in ast:
            if node["kind"] == "ident":
                text = code[node["start_byte"]:node["end_byte"]]

                if text == name and node["start_byte"] > pos:
                    used = True
                    break

        if not used:
            unused.append(name)

    return unused

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output/subgoalex.md"
    verifier = Lean4ServerScheduler(max_concurrent_requests=16, name='auto_sorrifier')
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Proof Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Raw Code\n\n")
        f.write("```lean\n")
        f.write(code)
        f.write("```\n\n")

        # Apply the same syntax correction pipeline as the main prover.
        code_corrected = SyntaxCorrector(code).correct_text()

        f.write("## Corrected Code\n\n")
        f.write("```lean\n")
        f.write(code_corrected)
        f.write("\n```\n\n")
        # code_corrected = code

        f.write("## Initial Verification\n\n")
        r = verifier.submit_all_request([dict(code=code_corrected, timeout=120)])
        r = verifier.get_all_request_outputs(r)[0]
        f.write(f"**Status:** {'✅ PASS' if r['pass'] else '❌ FAIL'}\n\n")

        if not r['pass'] and r.get('errors'):
            f.write("### Errors\n\n```bash\n")
            for e in r.get('errors', []):
                f.write(f"- **Line {e['pos']['line']}:** {e['data']}\n")
            f.write("\n```\n\n")

        f.write("## Sorrification Process\n\n")
        checker = AutoSorrifier(code_corrected, verifier, max_cycles=50)
        result = checker.fix_code()

        f.write("### Sorrified Result\n\n")
        f.write("```lean\n")
        f.write(result)
        f.write("\n```\n\n")
        final_code = result

        # f.write("## Auto Solver (ProofRepairer)\n\n")
        # from utils.hint_repair import ProofRepairer
        # repairer = ProofRepairer(result, verify_lean4_file)
        # final_code = repairer.repair_proof()

        # f.write("### Auto Solver Result\n\n")
        # f.write("```lean\n")
        # f.write(final_code)
        # f.write("\n```\n\n")

        # Verify final result
        final_check = verifier.submit_all_request([dict(code=final_code, timeout=120)])
        final_check = verifier.get_all_request_outputs(final_check)[0]
        f.write("## Final Verification\n\n")
        f.write(f"- **Pass:** {'Yes' if final_check['pass'] else 'No'}\n")
        f.write(f"- **Complete:** {'Yes' if final_check.get('complete', False) else 'No'}\n")

        # Warning-only: chỉ check unused have khi file đã compile không lỗi
        if final_check.get("pass"):
            try:
                print(f"Checking unused have(s) in final code...")
                # dump_ast_server đôi khi không đọc được file trong /tmp -> tạo temp ngay trong repl/
                with tempfile.NamedTemporaryFile(
                    "w",
                    suffix=".lean",
                    delete=False,
                    encoding="utf-8",
                    dir=REPL_DIR,
                ) as tf:
                    tf.write(final_code)
                    tmp_path = tf.name
                blocks = AST_DAEMON.get_ast(tmp_path)
                unused = find_unused_haves(blocks, final_code)
                if unused:
                    msg = f"WARNING: unused have(s): {sorted(unused)}"
                    print(msg)
                    f.write("\n## Warnings\n\n")
                    f.write(f"- {msg}\n")
            except Exception:
                pass
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        from utils.proof_state_extractor import extract_queries
        from retriever import retrieve

        f.write("## RAG Retrieval Analysis\n\n")

        queries = extract_queries(final_code)
        f.write(f"**Found {len(queries)} sorry(s)**\n\n")

        for i, q in enumerate(queries, 1):
            f.write(f"### Sorry #{i} (Line {q['line']})\n\n")
            f.write("#### Query\n\n```lean\n")
            f.write(q['raw'])
            f.write("\n```\n\n")

            # results = retrieve(q['raw'], k=10)
            # f.write(f"#### Retrieved {len(results)} Premises\n\n```lean\n")

            # for j, (premise, score) in enumerate(results, 1):
            #     f.write(f"**[{j}] {premise['full_name']}** (Similarity: {score:.4f})\n\n")
            #     premise_code = premise["code"]
            #     if len(premise_code) > 200:
            #         f.write(premise_code[:200] + "...")
            #     else:
            #         f.write(premise_code)
            #     f.write("\n")
            # f.write("\n```\n\n")

        verifier.close()

    print(f"Analysis complete! Report saved to: {output_file}")
print(f"Open the file to view the formatted results.")