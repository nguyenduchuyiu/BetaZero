from __future__ import annotations
import textwrap
from gammazero.core.nodes import ProofState
from gammazero.utils.lean_cmd import build_theorem

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

OUTPUT FORMAT EXAMPLE:
<think>
[Your thinking process goes here. Be concise and direct.]
</think>
```lean4
theorem my_theorem proposition := by
  [Your tactic sequence goes here. Be concise and direct.]
```
"""
).strip()


_SKELETON_INSTRUCTION = textwrap.dedent("""
You are a Dependency Skeleton Generator for a Lean 4 proof search tree.

Your task is to decompose the final goal into useful intermediate leaf obligations.
You are NOT solving the mathematical proof. You are producing a proof scaffold whose
only unsolved parts are explicitly named `have ... := sorry` leaf obligations.

CRITICAL CONSTRAINTS:

1. LEAF OBLIGATIONS ONLY:
   Every `sorry` must appear only in a named intermediate `have` statement.
   These `have` statements are the new subgoals for the search tree.

2. NO FINAL-GOAL SORRY:
   You are strictly forbidden from writing:
     `have h_final : <original final goal> := sorry`
   or any `have` whose proposition is syntactically identical or trivially equivalent
   to the original final goal with `:= sorry`.

3. FINAL ASSEMBLY MUST BE SORRY-FREE:
   The final goal must be closed by combining previously introduced hypotheses.
   The final assembly may use simple Lean proof terms/tactics such as:
     `exact ...`
     `apply ...`
     `constructor`
     `And.intro`
     `Or.inl`, `Or.inr`
     `Exists.intro`
     tuple notation `⟨..., ...⟩`
     `simpa using ...`
   But the final assembly itself must not contain `sorry`.

4. ALL IMPORTANT LEAVES MUST BE CONSUMED:
   Each generated leaf obligation should be useful for closing the final goal.
   Avoid producing decorative or unused facts.
   Prefer leaf obligations that correspond directly to missing proof pieces.

5. PROGRESS REQUIREMENT:
   Each `sorry` obligation must be a strict decomposition of the original goal.
   It should be simpler, narrower, or more local than the original goal.
   Do not restate the original goal under another name.

6. FLAT TOPOLOGY:
   Avoid branching search tactics such as `cases`, `rcases`, `induction`,
   `obtain`, or `by_cases` in the skeleton.
   If case analysis is mathematically needed, create a named leaf obligation
   that packages the needed result instead.

7. IF NO USEFUL DECOMPOSITION EXISTS:
   Output a minimal skeleton with one genuinely useful intermediate lemma if possible.
   If the goal is already atomic and cannot be decomposed, output `sorry`.

GOOD EXAMPLE:
```lean4
theorem my_theorem proposition := by
  have h1 : intermediate_prop_1 := sorry
  have h2 : intermediate_prop_2 := sorry
  exact final_assembly_using h1 h2
````

BAD EXAMPLE:

```lean4
theorem my_theorem proposition := by
  have h_final : proposition := sorry
  exact h_final
```

BAD EXAMPLE:

```lean4
theorem my_theorem proposition := by
  have h1 : intermediate_prop_1 := sorry
  have h_final : proposition := sorry
  exact h_final
```

Remember: the search tree solves the `sorry` leaf obligations. Your job is to make sure
that once those leaves are solved, the original goal closes automatically.

OUTPUT FORMAT EXAMPLE:
<think>
[Briefly explain the decomposition plan. Name the intermediate obligations and
why they are strictly simpler/useful for closing the final goal.]
</think>
```lean4
theorem my_theorem proposition := by
  have h1 : intermediate_prop_1 := sorry
  have h2 : intermediate_prop_2 := sorry
  exact final_assembly_using h1 h2
```
""").strip()


# _SKELETON_INSTRUCTION = textwrap.dedent("""
# You are a Subgoal Generator for a Search Tree in Lean 4. 
# Your task is to propose potential intermediate milestones to expand the search space, NOT to solve the problem.

# CRITICAL CONSTRAINTS:
# 1. DEFER VERIFICATION: You are explicitly forbidden from proving the subgoals. EVERY `have` statement MUST end with `:= sorry`. Even if a step is trivially true, you must delegate it to the search tree using `:= sorry`.
# 2. NO TERMINAL STATES: Do not attempt to close the final goal directly. You MUST wrap the final target in a `have` statement named `h_final` with `:= sorry`, and then close the block strictly with `exact h_final`.
# 3. FLAT TOPOLOGY ONLY: Branching tactics are incompatible with this search phase. NEVER use `cases`, `rcases`, `induction`, `obtain`, or `by_cases`. 

# Remember: You are generating an exploratory search node, not a finished proof. Incompleteness (using `sorry`) is the strict requirement for success.

# OUTPUT FORMAT EXAMPLE:
# <think>
# [Your thinking process goes here. Be concise and direct.]
# </think>
# ```lean4
# theorem my_theorem proposition := by
#   have h1 prop1 := sorry
#   have h2 prop2 := sorry
#   have h_final final_prop := sorry
#   exact h_final
# ```

# """).strip()

def _format_chatml_from_messages(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
    
    res = "\n".join(parts)
    if messages[-1]["role"] == "assistant":
        res = res.rsplit("\n<|im_end|>", 1)[0]
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
    elif action_type == "skeleton":
        instruction = _SKELETON_INSTRUCTION
    else:
        raise ValueError(action_type)
    
    full_system = instruction + '\n\n' + _OUTPUT_FORMAT_INSTRUCTION
    
    user_msg_content = _USER_BASE_INSTRUCTION + "\n" + _format_problem(state)
    if extra_rules:
        user_msg_content = extra_rules.strip() + "\n" + user_msg_content
        
    return [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_msg_content},
        {"role": "assistant", "content": "<think>\n"}
    ]

def build_prompt(state: ProofState, action_type: str, extra_rules: str = "") -> str:
    messages = build_messages(state, action_type, extra_rules)
    return _format_chatml_from_messages(messages)


def format_tactic_feedback_block(lean_code: str, lean_feedback: str) -> str:
    return (
        "FAILED TACTIC CODE:\n"
        "```lean4\n"
        f"{lean_code.strip()}\n"
        "```\n\n"
        "LEAN ERROR FEEDBACK:\n"
        f"{lean_feedback.strip()}"
    )


def build_tactic_retry_prompt(
    state: ProofState,
    feedback_blocks: list[str],
    *,
    max_feedbacks: int = 3,
) -> str:
    if not feedback_blocks:
        return build_prompt(state, "tactic")

    feedback_block = (
        "PREVIOUS TACTIC ATTEMPTS FAILED.\n"
        "Use the Lean feedback below to produce a NEW tactic proof for the same goal. "
        "Do not repeat the failed tactic. Preserve the exact theorem signature.\n\n"
        + "\n\n".join(feedback_blocks[-max_feedbacks:])
    )
    return build_prompt(state, "tactic", extra_rules=feedback_block)


def format_skeleton_feedback_block(lean_code: str, lean_feedback: str) -> str:
    return (
        "FAILED SKELETON CODE:\n"
        "```lean4\n"
        f"{lean_code.strip()}\n"
        "```\n\n"
        "LEAN ERROR FEEDBACK:\n"
        f"{lean_feedback.strip()}"
    )


def build_skeleton_retry_prompt(
    state: ProofState,
    feedback_blocks: list[str],
    *,
    max_feedbacks: int = 3,
) -> str:
    if not feedback_blocks:
        return build_prompt(state, "skeleton")

    feedback_block = (
        "PREVIOUS SKELETON ATTEMPTS FAILED.\n"
        "Use the Lean feedback below to produce a NEW skeleton for the same theorem. "
        "Do not repeat the failed skeleton. Preserve the exact theorem signature.\n\n"
        + "\n\n".join(feedback_blocks[-max_feedbacks:])
    )
    return build_prompt(state, "skeleton", extra_rules=feedback_block)


class SearchPromptBuilder:
    def __init__(self, *, max_skeleton_feedbacks: int = 3, max_tactic_feedbacks: int = 3):
        self.max_skeleton_feedbacks = max_skeleton_feedbacks
        self.max_tactic_feedbacks = max_tactic_feedbacks

    def build(
        self,
        state: ProofState,
        action_type: str,
        *,
        tactic_feedbacks: list[str] | None = None,
        skeleton_feedbacks: list[str] | None = None,
    ) -> str:
        if action_type == "tactic":
            return build_tactic_retry_prompt(
                state,
                tactic_feedbacks or [],
                max_feedbacks=self.max_tactic_feedbacks,
            )
        if action_type == "skeleton":
            return build_skeleton_retry_prompt(
                state,
                skeleton_feedbacks or [],
                max_feedbacks=self.max_skeleton_feedbacks,
            )
        return build_prompt(state, action_type)

    def format_tactic_feedback(self, lean_code: str, lean_feedback: str) -> str:
        return format_tactic_feedback_block(lean_code, lean_feedback)

    def format_skeleton_feedback(self, lean_code: str, lean_feedback: str) -> str:
        return format_skeleton_feedback_block(lean_code, lean_feedback)


def clean_prompt(text: str) -> str:
    return text.replace('\u00a0', ' ')
