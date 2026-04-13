def parse_proof_state_test(goal_str):
    s = (goal_str or "").strip()
    parts = s.split("⊢")
    if len(parts) > 1:
        ctx_raw = parts[0].strip()
        goal = parts[1].strip()
    else:
        ctx_raw, goal = "", s

    ctx_lines = []
    for line in ctx_raw.splitlines():
        if not line:
            continue
        
        # Heuristic from the fix
        if (line.startswith("  ") or not (":" in line)) and ctx_lines:
            ctx_lines[-1] = f"{ctx_lines[-1]} {line.strip()}"
        else:
            line_strip = line.strip()
            if ":" in line_strip and not line_strip.startswith("case"):
                ctx_lines.append(line_strip)

    ctx = "\n".join(ctx_lines)
    return ctx, goal

# Giả lập output từ Lean REPL với một giả thiết bị xuống dòng
repl_output = """x : ℝ
h₀ : 0 < x
h_mul_clear :
  3 * Real.log 2 * Real.log (Real.log x / (3 * Real.log 2)) =
  Real.log 2 * Real.log (Real.log x / Real.log 2)
⊢ Real.log ((t / 3) ^ 3) = Real.log t"""

ctx, goal = parse_proof_state_test(repl_output)
print("--- CONTEXT ---")
print(ctx)
print("--- GOAL ---")
print(goal)

if "h_mul_clear : 3 * Real.log 2" in ctx:
    print("\n✅ TEST PASSED: Hypothesis merged successfully.")
else:
    print("\n❌ TEST FAILED.")
