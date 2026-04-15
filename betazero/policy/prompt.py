from __future__ import annotations
import textwrap
from betazero.core.nodes import ProofState
from betazero.utils.lean_cmd import build_theorem

_OUTPUT_FORMAT_INSTRUCTION = textwrap.dedent(
"""
OUTPUT INSTRUCTIONS
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


_TACTIC_INSTRUCTION = textwrap.dedent("""
You are an elite Lean 4 Tactic Agent. Your objective is to close a goal.
You will be provided the [PROBLEM].

CRITICAL INSTRUCTIONS:
1. FILTER THE NOISE: The local context may contain irrelevant hypotheses. Inside the <think> tag, explicitly identify ONLY the hypotheses strictly necessary to prove the Goal. 
2. TACTIC REASONING: Sketch a short, direct sequence of tactics to close the goal.
3. FAIL FAST: If the goal is unprovable (due to a flawed premise), output `sorry`.
"""
).strip()


_SKELETON_INSTRUCTION = textwrap.dedent("""
You are a Subgoal Generator for a Search Tree in Lean 4. 
Your task is to propose potential intermediate milestones to expand the search space, NOT to solve the problem.

CRITICAL CONSTRAINTS:
1. DEFER VERIFICATION: You are explicitly forbidden from proving the subgoals. EVERY `have` statement MUST end with `:= sorry`. Even if a step is trivially true, you must delegate it to the search tree using `:= sorry`.
2. NO TERMINAL STATES: Do not attempt to close the final goal directly. You MUST wrap the final target in a `have` statement named `h_final` with `:= sorry`, and then close the block strictly with `exact h_final`.
3. FLAT TOPOLOGY ONLY: Branching tactics are incompatible with this search phase. NEVER use `cases`, `rcases`, `induction`, `obtain`, or `by_cases`. 

Remember: You are generating an exploratory search node, not a finished proof. Incompleteness (using `sorry`) is the strict requirement for success.

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


def _format_chatml_from_messages(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
    
    # Nếu tin nhắn cuối là assistant và rỗng hoặc là prefix, ta bỏ <|im_end|> cuối để model hoàn thiện
    # Nhưng trong ChatML chuẩn cho completion:
    res = "\n".join(parts)
    if messages[-1]["role"] == "assistant":
        # Bỏ <|im_end|> cuối cùng để model gõ tiếp
        res = res.rsplit("\n<|im_end|>", 1)[0] + "\n"
    return clean_prompt(res)


def _format_problem(state: ProofState) -> str:
    code = build_theorem(state, "sorry", name="my_theorem").rstrip()
    return (
        "[PROBLEM]\n"
        "```lean4\n"
        f"{code}\n"
        "```"
    )

def build_messages(state: ProofState, action_type: str, extra_rules: str = "") -> list[dict[str, str]]:
    if action_type == "tactic":
        instruction = _TACTIC_INSTRUCTION
        prefix = "\n>>> [MODE: TACTIC] SOLVE THE GOAL USING TACTICS (`by ...`).<<<\n"
    elif action_type == "skeleton":
        instruction = _SKELETON_INSTRUCTION
        prefix = "\n>>> [MODE: PLANNER] STRICTLY DEFER ALL PROOFS WITH `:= sorry`. NO TACTICS ALLOWED. <<<\n"
    else:
        raise ValueError(action_type)
    
    full_system = instruction + '\n\n' + _OUTPUT_FORMAT_INSTRUCTION
    
    user_msg_content = _USER_BASE_INSTRUCTION + "\n" + _format_problem(state)
    if extra_rules:
        user_msg_content = extra_rules.strip() + "\n" + user_msg_content
        
    return [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_msg_content},
        {"role": "assistant", "content": prefix}
    ]

def build_prompt(state: ProofState, action_type: str, extra_rules: str = "") -> str:
    messages = build_messages(state, action_type, extra_rules)
    return _format_chatml_from_messages(messages)

def build_tactic_self_correct_messages(
    state: ProofState,
    original_tactic: str,
    lean_feedback: str,
    sorrified_tactic: str,
) -> list[dict[str, str]]:
    full_system = _TACTIC_SELF_CORRECT_INSTRUCTION + '\n\n' + _OUTPUT_FORMAT_INSTRUCTION
    user_msg = (
        _USER_BASE_INSTRUCTION + "\n\n"
        + _format_problem(state) + "\n\n"
        + f"[PREVIOUS FAILED TACTIC]\n```lean4\n" + original_tactic.strip() + "\n```\n\n"
        + "[LEAN 4 COMPILER FEEDBACK]\n" + (lean_feedback.strip() or '(No clear error message)') + "\n\n"
        + "[SYNTAX-FIXED REFERENCE (Used sorry)]\n```lean4\n" + sorrified_tactic.strip() + "\n```"
    )
    return [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": "<think>\n"}
    ]

def build_tactic_self_correct_prompt(
    state: ProofState,
    original_tactic: str,
    lean_feedback: str,
    sorrified_tactic: str,
) -> str:
    messages = build_tactic_self_correct_messages(state, original_tactic, lean_feedback, sorrified_tactic)
    return _format_chatml_from_messages(messages)


def clean_prompt(text: str) -> str:
    return text.replace('\u00a0', ' ')

