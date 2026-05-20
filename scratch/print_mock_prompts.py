from __future__ import annotations

from gammazero.core import Action, ProofState
from gammazero.policy.prompt import (
    SearchPromptBuilder,
    build_skeleton_retry_prompt,
    build_tactic_retry_prompt,
)


def section(title: str, prompt: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(prompt)


def main() -> None:
    root = ProofState("h : True\nChild Sibling : Prop", "True")
    child = ProofState("h : True\nChild Sibling : Prop", "Child")
    sibling = ProofState("h : True\nChild Sibling : Prop", "Sibling")
    skeleton = Action(
        "skeleton",
        "mock skeleton",
        extracted_code=(
            "have h_child : Child := sorry\n"
            "have h_sibling : Sibling := sorry\n"
            "trivial"
        ),
        children=(child, sibling),
    )
    feedback = [
        "\n".join(
            [
                "FAILED CHECKED CODE:",
                "```lean4",
                "theorem my_theorem (h : True) : True := by",
                "  exact bad",
                "```",
                "",
                "LEAN ERROR FEEDBACK:",
                "unknown identifier 'bad'",
            ]
        )
    ]

    builder = SearchPromptBuilder()
    section("ROOT TACTIC RETRY", build_tactic_retry_prompt(root, feedback))
    section("ROOT SKELETON RETRY", build_skeleton_retry_prompt(root, feedback))
    section(
        "SUBGOAL TACTIC RETRY",
        builder.build_subgoal_tactic(
            root,
            skeleton,
            0,
            tactic_feedbacks=feedback,
        ),
    )
    section(
        "SUBGOAL SKELETON RETRY",
        builder.build_subgoal_skeleton(
            root,
            skeleton,
            0,
            skeleton_feedbacks=feedback,
        ),
    )


if __name__ == "__main__":
    main()
