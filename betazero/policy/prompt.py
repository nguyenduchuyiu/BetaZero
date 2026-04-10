from __future__ import annotations
import textwrap
from betazero.core.nodes import ProofState
from betazero.utils.lean_cmd import build_theorem

_SYSTEM_BASE_INSTRUCTION = textwrap.dedent(
    """
    You are an elite expert in Lean 4 theorem proving.
    CRITICAL RULES:
    1. You MUST use the exact theorem signature (name and arguments) provided in the [PROBLEM].
    2. OUTPUT FORMAT: After the </think> tag, you MUST output EXACTLY ONE valid ```lean4 ... ``` block containing your final answer. Do not add conversational text after the code block.
    3. Adjust the length of your <think> process to the complexity of the problem. If the problem is simple, a concise and direct breakdown is PERFECT. Do not artificially inflate the reasoning.
    """
).strip()

_USER_BASE_INSTRUCTION = textwrap.dedent(
    """
    This is the current state of the proof.
    You may only use the information in the problem statement below.
    """
).strip()

_TACTIC_INSTRUCTION = textwrap.dedent(
    """
    Write Lean 4 tactics to solve the goal based on the provided [PROBLEM].
    """
).strip()


_SKELETON_INSTRUCTION = textwrap.dedent("""
Your task is to translate informal mathematical plans into formal structural skeletons using `have` statements.

1. PLAN FIRST: Inside the <think> tag, write a step-by-step mathematical plan. Break the goal down logically.
2. FORMAL SKELETON: After closing </think>, output EXACTLY ONE valid ```lean4``` code block.
3. STRICT "HAVE" SYNTAX: You MUST explicitly declare the proposition type in EVERY `have` statement (e.g., `have h : a = b := ...`).
4. HANDLING SUBGOALS: 
   - For complex, multi-step subgoals, MUST leave them as `:= sorry`.
   - For trivial subgoals, you MAY close them using exactly ONE automated tactic: `:= by ring`, `:= by linarith`, `:= by norm_num`, `:= by omega`, `:= rfl`, or `:= trivial`. You may pass arguments (e.g., `:= by linarith [h1]`). 
   - Simple 1-hit term-mode applications (e.g., `:= ⟨x, y⟩` or `:= h1.trans h2`) are also allowed.
   - DO NOT write multi-line, chained tactics, or `calc` blocks.
   - DO NOT append comments (e.g., `-- explanation`) at the end of `have` lines.
5. NO BRANCHING IN SKELETON: DO NOT use `cases`, `induction`, `by_cases`, or `obtain` inside the skeleton. Extract them into flat lemmas (implications or functions).
   - If induction is needed, extract it into flat lemmas:
     `have h_base : P 0 := sorry`
     `have h_step : ∀ (k : ℕ), P k → P (k + 1) := sorry`
   - If case analysis is needed, extract the cases as implications:
     `have h_case1 : Case1 → Goal := sorry`
     `have h_case2 : Case2 → Goal := sorry`
6. CONNECTING SUBGOALS: Close the final goal using structural tactics (e.g., `exact`, `rw`, `Or.elim`, `Nat.recOn`) combining your `have` statements. The final goal may also be `sorry` if it requires complex reasoning.
""").strip()

_TACTIC_SELF_CORRECT_INSTRUCTION = textwrap.dedent(
  """
  Analyze the compiler feedback and the failed tactic. Write a complete Lean 4 tactic that fixes the error.
  You should use the [SYNTAX-FIXED REFERENCE] as a hint for the correct structure.

  ### EXAMPLES OF SELF-CORRECTION

  Example 1 (Error: Unknown identifier):

  [PROBLEM]

  ```lean4
  theorem my_theorem (n : ℕ) 
    : n + 0 = n := by
    sorry
  ```

  [PREVIOUS FAILED TACTIC]

  ```lean4
  theorem my_theorem (n : ℕ) 
    : n + 0 = n := by
    rw [add_zero_property]
  ```

  [LEAN 4 COMPILER FEEDBACK]
  error: unknown identifier 'add_zero_property'

  [SYNTAX-FIXED REFERENCE (Used sorry)]

  ```lean4
  theorem my_theorem (n : ℕ) 
    : n + 0 = n := by
    sorry
  ```

  # Example 1 OUTPUT

  <think>
  The compiler can't find 'add_zero_property'. Looking at the syntax-fixed reference, the correct lemma name is 'Nat.add_zero'.
  </think>

  ```lean4
  theorem my_theorem (n : ℕ) 
    : n + 0 = n := by
    rw [Nat.add_zero]
  ```
  
  END OF EXAMPLES
  Now, fix the failed tactic in the [PROBLEM] below using the same thought process and format.
  """
).strip()


def _format_chatml(system_msg: str, user_msg: str) -> str:
    full_prompt = (
        f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return clean_prompt(full_prompt)


def _format_problem(state: ProofState) -> str:
    code = build_theorem(state, "sorry", name="my_theorem").rstrip()
    return (
        "[PROBLEM]\n"
        "```lean4\n"
        f"{code}\n"
        "```"
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
        _USER_BASE_INSTRUCTION + "\n\n" +
        _format_problem(state)
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
        _USER_BASE_INSTRUCTION + "\n\n"
        + _format_problem(state) + "\n\n"
        + f"[PREVIOUS FAILED TACTIC]\n```lean4\n" + original_tactic.strip() + "\n```\n\n"
        + "[LEAN 4 COMPILER FEEDBACK]\n" + (lean_feedback.strip() or '(No clear error message)') + "\n\n"
        + "[SYNTAX-FIXED REFERENCE (Used sorry)]\n```lean4\n" + sorrified_tactic.strip() + "\n```"
    )
    return _format_chatml(full_system, user_msg)


def clean_prompt(text: str) -> str:
    return text.replace('\u00a0', ' ')
