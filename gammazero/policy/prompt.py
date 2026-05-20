from __future__ import annotations
import textwrap
import re

from gammazero.core.nodes import Action, ProofState
from gammazero.utils.lean_cmd import build_theorem
from gammazero.utils.scaffold import (
    render_single_target_scaffold,
    sorry_index_after_replacement,
    target_subgoal_label,
)

def _final_output_instructions(*, whole_scaffold: bool, skeleton: bool) -> str:
    code_scope = (
        "the WHOLE parent theorem scaffold"
        if whole_scaffold
        else "the exact theorem signature from [PROBLEM]"
    )
    rules = [
        "You MUST start with a <think>...</think> block explaining your approach.",
        f"After thinking, output EXACTLY ONE valid ```lean4 ... ``` block containing {code_scope}.",
    ]
    if whole_scaffold:
        rules.append(
            "Change ONLY the unique `sorry` subgoal; leave all `admit` placeholders perfectly intact."
        )
    if skeleton:
        rules.append(
            "Any new `sorry` must be a named leaf obligation; the final assembly must be sorry-free."
        )
    rules.append("Do not add any conversational text after the code block.")
    numbered_rules = "\n".join(
        f"{idx}. {rule}" for idx, rule in enumerate(rules, start=1)
    )
    return "\n".join(
        [
            "=========================================",
            "FINAL OUTPUT INSTRUCTIONS & FORMAT:",
            "The logs above are raw system outputs. You MUST NOT mimic their format.",
            "You must strictly follow this format:",
            "",
            numbered_rules,
            "",
            "EXAMPLE FORMAT:",
            "<think>",
            "[Your concise reasoning here]",
            "</think>",
            "```lean4",
            "[Your final Lean code here]",
            "```",
        ]
    )

_USER_BASE_INSTRUCTION = textwrap.dedent(
"""
This is the current state of the proof.
You may only use the information in the problem statement below.
"""
).strip()


_ROOT_TACTIC_INSTRUCTION = textwrap.dedent("""
You are an elite Lean 4 Tactic Agent. Your objective is to close a goal.
You will be provided the [PROBLEM].

CRITICAL INSTRUCTIONS:
1. FILTER THE NOISE: The local context may contain irrelevant hypotheses. Inside the <think> tag, explicitly identify ONLY the hypotheses strictly necessary to prove the Goal. 
2. TACTIC REASONING: Sketch a short, direct sequence of tactics to close the goal.
3. PLACEHOLDER BAN: Your Lean proof body must not contain `sorry` or `admit`.
"""
).strip()


_SUBGOAL_TACTIC_INSTRUCTION = textwrap.dedent("""
You are an elite Lean 4 Subgoal Tactic Agent.

You will be given the full parent proof scaffold, not an isolated theorem for
the subgoal. Exactly one placeholder is written as `sorry`; that is the current
subgoal you must solve. Other sibling placeholders are written as `admit`; they
are intentionally left for other search nodes.
The user message names the target in a TARGET SUBGOAL section.

CRITICAL INSTRUCTIONS:
1. Solve ONLY the named TARGET SUBGOAL, which is also the unique `sorry`
   placeholder. Do not solve or edit any `admit`.
2. Your final Lean code block must contain the WHOLE parent theorem scaffold
   from [PROBLEM], with only the unique `sorry` placeholder replaced by your
   proof.
3. Do not change any code outside the subgoal marked by the unique `sorry`.
   Keep every sibling `admit` exactly as an `admit`.
4. The replacement proof for the target subgoal must not contain `sorry` or `admit` in tactics.
5. Use the surrounding scaffold to preserve Lean's original elaboration context.
""").strip()


_SUBGOAL_SKELETON_INSTRUCTION = textwrap.dedent("""
You are a Lean 4 Subgoal Skeleton Generator.

You will be given the full parent proof scaffold, not an isolated theorem for
the subgoal. Exactly one placeholder is written as `sorry`; that is the current
subgoal you must decompose. Other sibling placeholders are written as `admit`;
they are intentionally left for other search nodes.
The user message names the target in a TARGET SUBGOAL section.

CRITICAL INSTRUCTIONS:
1. Replace ONLY the named TARGET SUBGOAL, which is also the unique `sorry`
   placeholder, with a mini-skeleton proof.
2. Your final Lean code block must contain the WHOLE parent theorem scaffold
   from [PROBLEM], with only the unique `sorry` placeholder replaced.
3. Do not change any code outside the subgoal marked by the unique `sorry`.
   Keep every sibling `admit` exactly as an `admit`.
4. Use the surrounding scaffold to preserve Lean's original elaboration context.

MINI-SKELETON CONSTRAINTS:

1. LEAF OBLIGATIONS ONLY:
   Every new `sorry` in your replacement must appear only in a named
   intermediate `have ... := sorry` statement. These named `have`s are the new
   child subgoals for the search tree.

2. NO TARGET-GOAL SORRY:
   You are strictly forbidden from replacing the target with:
     `have h_final : <target subgoal proposition> := sorry`
   or any `have` whose proposition is syntactically identical or trivially
   equivalent to the target subgoal with `:= sorry`.

3. TARGET ASSEMBLY MUST BE SORRY-FREE:
   After introducing the new leaf obligations, the replacement mini-skeleton
   must close the target subgoal by combining those leaves and the surrounding
   local context. The final assembly inside the target replacement may use
   simple Lean proof terms/tactics such as:
     `exact ...`
     `apply ...`
     `constructor`
     `And.intro`
     `Or.inl`, `Or.inr`
     `Exists.intro`
     tuple notation `⟨..., ...⟩`
     `simpa using ...`
   But the target assembly itself must not contain `sorry`.

4. ALL IMPORTANT LEAVES MUST BE CONSUMED:
   Each generated leaf obligation should be useful for closing the target
   subgoal. Avoid decorative or unused facts. Prefer leaves that correspond
   directly to missing proof pieces.

5. PROGRESS REQUIREMENT:
   Each new `sorry` obligation must be a strict decomposition of the target
   subgoal. It should be simpler, narrower, or more local than the target.
   Do not restate the target subgoal under another name.

6. FLAT TOPOLOGY:
   Avoid branching search tactics such as `cases`, `rcases`, `induction`,
   `obtain`, or `by_cases` inside the mini-skeleton. If case analysis is
   mathematically needed, create a named leaf obligation that packages the
   needed result instead.

7. IF NO USEFUL DECOMPOSITION EXISTS:
   Output a minimal mini-skeleton with one genuinely useful intermediate lemma
   if possible. Do not leave the target subgoal as a naked `sorry`; if no valid
   decomposition exists, produce the best named leaf obligation and close the
   target assembly from that leaf.

BAD EXAMPLE:
```lean4
theorem my_theorem proposition := by
  have h1 : intermediate_prop_1 := admit
  have h2 : target_subgoal_prop := by
    have h_final : target_subgoal_prop := sorry
    exact h_final
  have h3 : intermediate_prop_3 := admit
  exact final_assembly
```
""").strip()


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
   Do not leave the final goal as a naked `sorry`; if no valid decomposition
   exists, produce the best named leaf obligation and close the final assembly
   from that leaf.

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

""").strip()


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


def _build_chatml_prompt(system: str, user: str, assistant_prefill: str = "<think>\n") -> str:
    return _format_chatml_from_messages(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_prefill},
        ]
    )


def _format_problem(state: ProofState) -> str:
    code = (
        render_single_target_scaffold(state.scaffold_code, state.target_index).rstrip()
        if state.scaffold_code
        else build_theorem(state, "sorry", name="my_theorem").rstrip()
    )
    return (
        "[PROBLEM]\n"
        "```lean4\n"
        f"{code}\n"
        "```"
    )


def render_subgoal_tactic_code(
    parent_state: ProofState,
    skeleton: Action,
    target_child_index: int,
    child_state: ProofState | None = None,
) -> str:
    if child_state is not None and child_state.scaffold_code:
        return render_single_target_scaffold(
            child_state.scaffold_code,
            child_state.target_index,
        ).rstrip()

    parts = re.split(r"\bsorry\b", skeleton.extracted_code)
    sorry_count = len(parts) - 1
    if sorry_count != len(skeleton.children) or target_child_index >= sorry_count:
        if parent_state.scaffold_code:
            return render_single_target_scaffold(
                parent_state.scaffold_code,
                parent_state.target_index,
            ).rstrip()
        return build_theorem(parent_state, skeleton.extracted_code, name="my_theorem").rstrip()

    body = parts[0]
    for i in range(sorry_count):
        body += "sorry" if i == target_child_index else "by admit"
        body += parts[i + 1]
    if parent_state.scaffold_code:
        from gammazero.utils.scaffold import replace_sorry_at

        scaffold = replace_sorry_at(parent_state.scaffold_code, parent_state.target_index, body)
        target_index = sorry_index_after_replacement(
            parent_state.scaffold_code,
            parent_state.target_index,
            body,
            target_child_index,
        )
        return render_single_target_scaffold(scaffold, target_index).rstrip()
    return build_theorem(parent_state, body, name="my_theorem").rstrip()


def _format_subgoal_tactic_problem(
    parent_state: ProofState,
    skeleton: Action,
    target_child_index: int,
    child_state: ProofState | None = None,
) -> str:
    code = render_subgoal_tactic_code(parent_state, skeleton, target_child_index, child_state)
    target_label = target_subgoal_label(
        skeleton.extracted_code,
        target_child_index,
        target_kind="skeleton_child",
    )
    target_goal = (
        skeleton.children[target_child_index].goal
        if 0 <= target_child_index < len(skeleton.children)
        else ""
    )
    return (
        "[PROBLEM]\n"
        "```lean4\n"
        f"{code}\n"
        "```\n\n"
        "[TARGET SUBGOAL]\n"
        f"name: {target_label}\n"
        f"child_index: {target_child_index}\n"
        f"goal: {target_goal}\n\n"
        f"Solve exactly the `sorry` for target subgoal `{target_label}`. "
        "The `admit` placeholders are sibling subgoals."
    )


def _format_subgoal_skeleton_problem(
    parent_state: ProofState,
    skeleton: Action,
    target_child_index: int,
    child_state: ProofState | None = None,
) -> str:
    code = render_subgoal_tactic_code(parent_state, skeleton, target_child_index, child_state)
    target_label = target_subgoal_label(
        skeleton.extracted_code,
        target_child_index,
        target_kind="skeleton_child",
    )
    target_goal = (
        skeleton.children[target_child_index].goal
        if 0 <= target_child_index < len(skeleton.children)
        else ""
    )
    return (
        "[PROBLEM]\n"
        "```lean4\n"
        f"{code}\n"
        "```\n\n"
        "[TARGET SUBGOAL]\n"
        f"name: {target_label}\n"
        f"child_index: {target_child_index}\n"
        f"goal: {target_goal}\n\n"
        f"Decompose exactly the `sorry` for target subgoal `{target_label}` into a mini-skeleton. "
        "The `admit` placeholders are sibling subgoals."
    )


def _root_system_instruction(action_type: str) -> str:
    if action_type == "tactic":
        instruction = _ROOT_TACTIC_INSTRUCTION
    elif action_type == "skeleton":
        instruction = _SKELETON_INSTRUCTION
    else:
        raise ValueError(action_type)

    return instruction


def _root_user_message(state: ProofState, action_type: str, extra_rules: str = "") -> str:
    user_msg_content = _USER_BASE_INSTRUCTION + "\n" + _format_problem(state)
    if extra_rules:
        user_msg_content = user_msg_content + "\n\n" + extra_rules.strip()
    user_msg_content += "\n\n" + _final_output_instructions(
        whole_scaffold=False,
        skeleton=action_type == "skeleton",
    )
    return user_msg_content


def build_messages(state: ProofState, action_type: str, extra_rules: str = "") -> list[dict[str, str]]:
    """Return structured ChatML messages for callers that need message-level access."""
    return [
        {"role": "system", "content": _root_system_instruction(action_type)},
        {"role": "user", "content": _root_user_message(state, action_type, extra_rules)},
        {"role": "assistant", "content": "<think>\n"},
    ]


def build_prompt(state: ProofState, action_type: str, extra_rules: str = "") -> str:
    return _build_chatml_prompt(
        _root_system_instruction(action_type),
        _root_user_message(state, action_type, extra_rules),
    )


def build_subgoal_tactic_prompt(
    parent_state: ProofState,
    skeleton: Action,
    target_child_index: int,
    feedback_blocks: list[str] | None = None,
    child_state: ProofState | None = None,
    *,
    max_feedbacks: int = 3,
) -> str:
    full_system = _SUBGOAL_TACTIC_INSTRUCTION
    user_msg_content = _USER_BASE_INSTRUCTION + "\n" + _format_subgoal_tactic_problem(
        parent_state,
        skeleton,
        target_child_index,
        child_state,
    )
    if feedback_blocks:
        feedback_block = (
            "PREVIOUS SUBGOAL TACTIC ATTEMPTS FAILED.\n"
            "Use the Lean feedback below to produce a NEW parent scaffold for the same `sorry`. "
            "Do not repeat the failed tactic.\n\n"
            + "\n\n".join(feedback_blocks[-max_feedbacks:])
        )
        user_msg_content += "\n\n" + feedback_block
    user_msg_content += "\n\n" + _final_output_instructions(
        whole_scaffold=True,
        skeleton=False,
    )
    return _build_chatml_prompt(full_system, user_msg_content)


def build_subgoal_skeleton_prompt(
    parent_state: ProofState,
    skeleton: Action,
    target_child_index: int,
    feedback_blocks: list[str] | None = None,
    child_state: ProofState | None = None,
    *,
    max_feedbacks: int = 3,
) -> str:
    full_system = _SUBGOAL_SKELETON_INSTRUCTION
    user_msg_content = _USER_BASE_INSTRUCTION + "\n" + _format_subgoal_skeleton_problem(
        parent_state,
        skeleton,
        target_child_index,
        child_state,
    )
    if feedback_blocks:
        feedback_block = (
            "PREVIOUS SUBGOAL SKELETON ATTEMPTS FAILED.\n"
            "Use the Lean feedback below to produce a NEW parent scaffold for the same `sorry`. "
            "Do not repeat the failed mini-skeleton.\n\n"
            + "\n\n".join(feedback_blocks[-max_feedbacks:])
        )
        user_msg_content += "\n\n" + feedback_block
    user_msg_content += "\n\n" + _final_output_instructions(
        whole_scaffold=True,
        skeleton=True,
    )
    return _build_chatml_prompt(full_system, user_msg_content)


def format_tactic_feedback_block(checked_code: str, lean_feedback: str) -> str:
    return (
        "FAILED CHECKED CODE:\n"
        "```lean4\n"
        f"{checked_code.strip()}\n"
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


def format_skeleton_feedback_block(checked_code: str, lean_feedback: str) -> str:
    return (
        "FAILED CHECKED CODE:\n"
        "```lean4\n"
        f"{checked_code.strip()}\n"
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

    def build_subgoal_tactic(
        self,
        parent_state: ProofState,
        skeleton: Action,
        target_child_index: int,
        *,
        child_state: ProofState | None = None,
        tactic_feedbacks: list[str] | None = None,
    ) -> str:
        return build_subgoal_tactic_prompt(
            parent_state,
            skeleton,
            target_child_index,
            tactic_feedbacks or [],
            child_state=child_state,
            max_feedbacks=self.max_tactic_feedbacks,
        )

    def build_subgoal_skeleton(
        self,
        parent_state: ProofState,
        skeleton: Action,
        target_child_index: int,
        *,
        child_state: ProofState | None = None,
        skeleton_feedbacks: list[str] | None = None,
    ) -> str:
        return build_subgoal_skeleton_prompt(
            parent_state,
            skeleton,
            target_child_index,
            skeleton_feedbacks or [],
            child_state=child_state,
            max_feedbacks=self.max_skeleton_feedbacks,
        )

    def format_tactic_feedback(self, checked_code: str, lean_feedback: str) -> str:
        return format_tactic_feedback_block(checked_code, lean_feedback)

    def format_skeleton_feedback(self, checked_code: str, lean_feedback: str) -> str:
        return format_skeleton_feedback_block(checked_code, lean_feedback)


def clean_prompt(text: str) -> str:
    return text.replace('\u00a0', ' ')
