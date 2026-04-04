import textwrap
from betazero.core import ProofState

_SYSTEM_INSTRUCTION = textwrap.dedent("""\
You are an elite expert in Lean 4 theorem proving.
Your task is to advance or solve the given mathematical state.

STRICT OUTPUT RULES:
1. You MUST think step-by-step inside <think>...</think> tags first.
2. Then output EXACTLY ONE ```lean4 code block.
3. No conversational explanations outside the tags.
""").strip()

_TACTIC_INSTRUCTION = textwrap.dedent("""\
Write Lean 4 tactics to solve the goal.

Output Example:
<think>
Brief reasoning about which tactics to apply.
</think>
```lean4
theorem solve_current_goal (hypotheses...) : goal := by
    tactic_step_1
    tactic_step_2
```
""").strip()

_SKELETON_INSTRUCTION = textwrap.dedent("""

Write a modular Lean 4 proof skeleton. Use sorry for unresolved subgoals.

Output Example:
<think>
Break the goal into intermediate steps and prove them separately.
</think>
```lean4
theorem solve_current_goal (hypotheses...) : goal := by
    have h1 : intermediate_step_1 := by
        sorry
    have h2 : intermediate_step_2 := by
        sorry
    exact final_step
```
""").strip()

_TACTIC_SELF_CORRECT_INSTRUCTION = textwrap.dedent("""

Analyze the compiler feedback. Write a CORRECTED Lean 4 tactic to replace the failed one.

Output Example:
<think>
Analyze why the previous tactic failed and how to fix it.
</think>
```lean4
theorem solve_current_goal (hypotheses...) : goal := by
    corrected_tactic_step_1
    corrected_tactic_step_2
```
""").strip()

def _format_chatml(system_msg: str, user_msg: str) -> str:
    return (
        f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )

def build_prompt(state: ProofState, action_type: str) -> str:
    if action_type == "tactic":
        instruction = _TACTIC_INSTRUCTION
    elif action_type == "skeleton":
        instruction = _SKELETON_INSTRUCTION
    else:
        raise ValueError(action_type)
    user_msg = (
        "[CONTEXT]\n" + state.context.strip() + "\n\n" +
        "[GOAL]\n" + state.goal.strip() + "\n\n" +
        "[INSTRUCTION]\n" + instruction
    )
    return _format_chatml(_SYSTEM_INSTRUCTION, user_msg)

def build_tactic_self_correct_prompt(
    state: ProofState,
    original_tactic: str,
    lean_feedback: str,
    sorrified_tactic: str,
    ) -> str:
    user_msg = (
        "[CONTEXT]\n" + state.context.strip() + "\n\n" +
        "[GOAL]\n" + state.goal.strip() + "\n\n" +
        "[PREVIOUS FAILED TACTIC]\nlean4\n" + original_tactic.strip() + "\n\n\n" +
        "[LEAN 4 COMPILER FEEDBACK]\n" + (lean_feedback.strip() or '(No clear error message)') + "\n\n" +
        "[SYNTAX-FIXED REFERENCE (Used sorry)]\nlean4\n" + sorrified_tactic.strip() + "\n\n\n" +
        "[INSTRUCTION]\n" + _TACTIC_SELF_CORRECT_INSTRUCTION
    )
    return _format_chatml(_SYSTEM_INSTRUCTION, user_msg)
