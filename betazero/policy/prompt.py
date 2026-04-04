import textwrap
from betazero.core import ProofState

_SYSTEM_INSTRUCTION = textwrap.dedent("""\
    You are an elite expert in Lean 4 theorem proving.
    Your task is to advance or solve the given mathematical state.

    STRICT OUTPUT RULES:
    1. REASONING: You MUST think step-by-step inside <think>...</think> tags first.
    2. CODE: Output EXACTLY ONE code block starting with ```lean4 and ending with ```.
    3. NO YAPPING: Absolutely NO conversational text, greetings, or explanations outside the tags.
""").strip()


def build_prompt(state: ProofState, action_type: str) -> str:
    """Builds a highly constrained prompt for generating Tactics or Proof Skeletons."""
    
    if action_type == "tactic":
        action_directive = "Write ONLY the Lean 4 tactic steps (lines that go under `by`) to advance the goal."
    else:
        action_directive = "Write a modular Lean 4 proof skeleton. Use `sorry` for unresolved subgoals."

    return textwrap.dedent(f"""\
        {_SYSTEM_INSTRUCTION}

        [CONTEXT]
        {state.context.strip()}

        [GOAL]
        {state.goal.strip()}

        [INSTRUCTION]
        {action_directive}
    """)


def build_tactic_self_correct_prompt(
    state: ProofState,
    original_tactic: str,
    lean_feedback: str,
    sorrified_tactic: str,
) -> str:
    """Forces the LLM to analyze its own mistake and output a corrected tactic."""
    
    return textwrap.dedent(f"""\
        {_SYSTEM_INSTRUCTION}

        [CONTEXT]
        {state.context.strip()}

        [GOAL]
        {state.goal.strip()}

        [PREVIOUS FAILED TACTIC]
        ```lean4
        {original_tactic.strip()}
        ```

        [LEAN 4 COMPILER FEEDBACK]
        {lean_feedback.strip() or '(No clear error message)'}

        [SYNTAX-FIXED REFERENCE (Used sorry)]
        ```lean4
        {sorrified_tactic.strip()}
        ```

        [INSTRUCTION]
        Analyze the compiler feedback. Write a CORRECTED Lean 4 tactic to replace the failed one. 
        DO NOT wrap it in `example` or `theorem`. Follow the STRICT OUTPUT RULES.
    """)