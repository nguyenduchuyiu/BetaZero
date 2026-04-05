import textwrap
from betazero.core import ProofState
from betazero.search.sorrifier.dependency_analyzer import SHARED_EXPR_ANALYZER
from betazero.env.expr_parser import get_lean_expr_tree

_SYSTEM_BASE_INSTRUCTION = textwrap.dedent("""\
You are an elite expert in Lean 4 theorem proving.
Your task is to advance or solve the given mathematical state.

STRICT OUTPUT RULES:
1. You MUST think step-by-step inside <think>...</think> tags first.
2. Then output EXACTLY ONE ```lean4 code block.
3. No conversational explanations outside the tags.
""").strip()

_USER_BASE_INSTRUCTION = textwrap.dedent("""\
This is the current state of the proof.
You may only use the information in the [CONTEXT] and [GOAL] to solve the problem.
""").strip()

_TACTIC_INSTRUCTION = textwrap.dedent("""\
Write Lean 4 tactics to solve the goal based on the provided [CONTEXT] and [GOAL].

CRITICAL: Do NOT copy the literal names from the example below. You MUST use the actual theorem name, actual variables, and actual goal from the current state.

Output Format Example:
<think>
Brief reasoning about which tactics to apply.
</think>
```lean4
theorem <ACTUAL_THEOREM_NAME> (<ACTUAL_VARIABLES>) : <ACTUAL_GOAL> := by
    <your_actual_tactic_1>
    <your_actual_tactic_2>
```
""").strip()

_SKELETON_INSTRUCTION = textwrap.dedent("""\
Write a modular Lean 4 proof skeleton based on the provided [CONTEXT] and [GOAL]. Use sorry for unresolved subgoals.

CRITICAL: Do NOT output literal strings like "solve_current_goal". You MUST use the actual theorem signature from the state.

Output Format Example:
<think>
Break the goal into intermediate steps and prove them separately.
</think>
```lean4
theorem <ACTUAL_THEOREM_NAME> (<ACTUAL_VARIABLES>) : <ACTUAL_GOAL> := by
    have h1 : <actual_intermediate_step_1> := by
        sorry
    have h2 : <actual_intermediate_step_2> := by
        sorry
    exact <actual_final_step>
```
""").strip()

_TACTIC_SELF_CORRECT_INSTRUCTION = textwrap.dedent("""

Analyze the compiler feedback. Write a CORRECTED Lean 4 tactic to replace the failed one.

CRITICAL: Use the actual theorem signature from the state. Do NOT use placeholder names from the example.

Output Format Example:
<think>
Analyze why the previous tactic failed and how to fix it.
</think>
```lean4
theorem <ACTUAL_THEOREM_NAME> (<ACTUAL_VARIABLES>) : <ACTUAL_GOAL> := by
    <corrected_actual_tactic_1>
    <corrected_actual_tactic_2>
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

    full_system = _SYSTEM_BASE_INSTRUCTION + '\n\n' + instruction
    user_msg = (
        "[CONTEXT]\n" + state.context.strip() + "\n\n" +
        "[GOAL]\n" + state.goal.strip()
    )
    return _format_chatml(full_system, user_msg)

def build_tactic_self_correct_prompt(
    state: ProofState,
    original_tactic: str,
    lean_feedback: str,
    sorrified_tactic: str,
    ) -> str:
    
    full_system = _SYSTEM_BASE_INSTRUCTION + '\n\n' + _TACTIC_SELF_CORRECT_INSTRUCTION
    user_msg = (
        "[CONTEXT]\n" + state.context.strip() + "\n\n" +
        "[GOAL]\n" + state.goal.strip() + "\n\n" +
        "[PREVIOUS FAILED TACTIC]\nlean4\n" + original_tactic.strip() + "\n\n\n" +
        "[LEAN 4 COMPILER FEEDBACK]\n" + (lean_feedback.strip() or '(No clear error message)') + "\n\n" +
        "[SYNTAX-FIXED REFERENCE (Used sorry)]\nlean4\n" + sorrified_tactic.strip()
    )
    return _format_chatml(full_system, user_msg)
