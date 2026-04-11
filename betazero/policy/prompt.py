from __future__ import annotations
import textwrap
from betazero.core.nodes import ProofState
from betazero.utils.lean_cmd import build_theorem

_OUTPUT_FORMAT_INSTRUCTION = textwrap.dedent(
    """
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
You are a Subgoal Generator for a Search Tree in Lean 4. 
Your task is to propose potential intermediate milestones to expand the search space, NOT to solve the problem.

CRITICAL CONSTRAINTS:
1. DEFER VERIFICATION: You are explicitly forbidden from proving the subgoals. EVERY `have` statement MUST end with `:= sorry`. Even if a step is trivially true, you must delegate it to the search tree using `:= sorry`.
2. NO TERMINAL STATES: Do not attempt to close the final goal. The proof must remain deliberately incomplete. Connect your subgoals and close the code block with a final `exact sorry` or just `sorry`.
3. FLAT TOPOLOGY ONLY: Branching tactics are incompatible with this search phase. NEVER use `cases`, `rcases`, `induction`, `obtain`, or `by_cases`. 

Remember: You are generating an exploratory search node, not a finished proof. Incompleteness (using `sorry`) is the strict requirement for success.

### EXAMPLE

[INPUT]
```lean4
import Mathlib
open Nat
theorem div_six (n : ℕ) : 6 ∣ n^3 - n := by sorry
````

[EXPECTED OUTPUT]
<think>
(Your thinking process here)
</think>
```lean4
theorem div_six (n : ℕ) : 6 ∣ n^3 - n := by
  have h2 : 2 ∣ n^3 - n := sorry
  have h3 : 3 ∣ n^3 - n := sorry
  have h_coprime : Nat.Coprime 2 3 := sorry
  have h_combine : 2 * 3 ∣ n^3 - n := sorry
  have h_final : 6 ∣ n^3 - n := sorry
  exact h_final
```

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
    full_system = instruction + '\n\n' + _OUTPUT_FORMAT_INSTRUCTION
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
    full_system = _TACTIC_SELF_CORRECT_INSTRUCTION + '\n\n' + _OUTPUT_FORMAT_INSTRUCTION
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
