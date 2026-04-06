from __future__ import annotations
import textwrap
from betazero.core.nodes import ProofState

_SYSTEM_BASE_INSTRUCTION = textwrap.dedent("""\
You are an elite expert in Lean 4 theorem proving.
Your task is to advance or solve the given mathematical state.
""").strip()

_USER_BASE_INSTRUCTION = textwrap.dedent("""\
This is the current state of the proof.
You may only use the information in the following [CONTEXT] and [GOAL].
""").strip()

_TACTIC_INSTRUCTION = textwrap.dedent("""\
Write Lean 4 tactics to solve the goal based on the provided [CONTEXT] and [GOAL].

### EXAMPLES OF EXPECTED OUTPUT

Example 1:

[CONTEXT]
p q : Prop
hp : p
hq : q

[GOAL]
p ∧ q

# Example 1 OUTPUT

<think>
The goal is a conjunction. I have both individual components in the context. I can use the constructor tactic.
</think>
```lean4
theorem solve_conj (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor
  · exact hp
  · exact hq
```
Example 2:

[CONTEXT]
n : ℕ

[GOAL]
n + 0 = n

# Example 2 OUTPUT
<think>
This is a fundamental property of natural numbers in Lean. The rw tactic with Nat.add_zero will solve it.
</think>
```lean4
theorem add_zero_id (n : ℕ) : n + 0 = n := by
  rw [Nat.add_zero]
```
""").strip()

_SKELETON_INSTRUCTION = textwrap.dedent("""\
Write a modular Lean 4 proof skeleton. Break the goal into logical intermediate steps using 'have'.
### EXAMPLES OF EXPECTED OUTPUT

Example 1:

[CONTEXT]
a b c : ℕ

[GOAL]
(a + b) + c = a + (b + c)

# Example 1 OUTPUT
<think>
I will prove this by showing the left side equals the right side using associativity.
</think>
```lean4
theorem add_assoc_skeleton (a b c : ℕ) : (a + b) + c = a + (b + c) := by
  have h1 : (a + b) + c = a + b + c := by
    sorry
  have h2 : a + (b + c) = a + b + c := by
    sorry
  rw [h1, h2]
```

Example 2:

[CONTEXT]
p q : Prop

[GOAL]
p ↔ q

# Example 2 OUTPUT
<think>
An iff goal requires proving two directions: p → q and q → p.
</think>
```lean4
theorem iff_skeleton (p q : Prop) : p ↔ q := by
  constructor
  · intro hp
    sorry
  · intro hq
    sorry
```
""").strip()


_TACTIC_SELF_CORRECT_INSTRUCTION = textwrap.dedent("""\
Analyze the compiler feedback and the failed tactic. Write a complete Lean 4 tactic that fixes the error.
You should use the [SYNTAX-FIXED REFERENCE] as a hint for the correct structure.

### EXAMPLES OF SELF-CORRECTION

Example 1 (Error: Unknown identifier):

[CONTEXT]
n : ℕ

[GOAL]
n + 0 = n

[PREVIOUS FAILED TACTIC]
```lean4
theorem add_zero_fix (n : ℕ) : n + 0 = n := by
  rw [add_zero_property]
```
  
[LEAN 4 COMPILER FEEDBACK]
error: unknown identifier 'add_zero_property'

[SYNTAX-FIXED REFERENCE (Used sorry)]
```lean4
theorem add_zero_fix (n : ℕ) : n + 0 = n := by
  sorry
```

# Example 1 OUTPUT
<think>
The compiler can't find 'add_zero_property'. Looking at the syntax-fixed reference, the correct lemma name is 'Nat.add_zero'.
</think>
```lean4
theorem add_zero_fix (n : ℕ) : n + 0 = n := by
  rw [Nat.add_zero]
```
""").strip()

def _format_chatml(system_msg: str, user_msg: str) -> str:
    full_prompt = (
        f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    return clean_prompt(full_prompt)

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
        _USER_BASE_INSTRUCTION + "\n\n" +
        "[CONTEXT]\n" + state.context.strip() + "\n\n" +
        "[GOAL]\n" + state.goal.strip() + "\n\n" +
        f"[PREVIOUS FAILED TACTIC]\n```lean4\n" + original_tactic.strip() + "\n```\n\n" +
        "[LEAN 4 COMPILER FEEDBACK]\n" + (lean_feedback.strip() or '(No clear error message)') + "\n\n" +
        "[SYNTAX-FIXED REFERENCE (Used sorry)]\n```lean4\n" + sorrified_tactic.strip() + "\n```"
    )
    return _format_chatml(full_system, user_msg)

def clean_prompt(text: str) -> str:
    # Thay thế NBSP (khoảng trắng lạ) bằng khoảng trắng chuẩn ASCII 32
    return text.replace('\u00a0', ' ')

# if __name__ == "__main__":
#     state = ProofState(
#         context="n : ℕ",
#         goal="n + 0 = n",
#     )
#     original_tactic = "theorem add_zero_fix (n : ℕ) : n + 0 = n := by\n  rw [add_zero_property]"
#     lean_feedback = "error: unknown identifier 'add_zero_property'"
#     sorrified_tactic = "theorem add_zero_fix (n : ℕ) : n + 0 = n := by\n  sorry"
#     print("-" * 100)
#     print("SELF-CORRECT PROMPT")
#     print("-" * 100)
#     print(build_tactic_self_correct_prompt(state, original_tactic, lean_feedback, sorrified_tactic))
#     print("-" * 100)
#     print("SKELETON PROMPT")
#     print("-" * 100)
#     print(build_prompt(state, "skeleton"))
#     print("-" * 100)
#     print("TACTIC PROMPT")
#     print("-" * 100)
#     print(build_prompt(state, "tactic"))
#     print("-" * 100)