import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gammazero.policy.output_parser import (
    INVALID_SKELETON_FEEDBACK,
    SUBGOAL_EXTRACTION_FEEDBACK,
    TRUNCATED_THINK_FEEDBACK,
    explain_empty_lean_code,
    get_lean_code,
    get_subgoal_skeleton_code,
    get_subgoal_tactic_code,
    validate_skeleton_replacement,
)


def test_get_lean_code_requires_final_fence():
    raw = """<think>
done
</think>
```lean4
theorem my_theorem : True := by
  trivial
```
unfinished trailing analysis
"""

    assert get_lean_code(raw) == ""


def test_explain_empty_lean_code_reports_truncated_thinking_output():
    raw = """<think>
This is getting too long.
</think>
```lean4
theorem my_theorem : True := by
  trivial
"""

    assert explain_empty_lean_code(raw) == TRUNCATED_THINK_FEEDBACK


def test_explain_empty_lean_code_reports_subgoal_extraction_failure():
    raw = """```lean4
theorem my_theorem : True := by
  trivial
```"""

    assert explain_empty_lean_code(raw, subgoal=True) == SUBGOAL_EXTRACTION_FEEDBACK


def test_explain_empty_lean_code_prefers_finish_reason_truncation_for_subgoals():
    raw = """```lean4
theorem my_theorem : True := by
  trivial
```"""

    assert (
        explain_empty_lean_code(raw, subgoal=True, finish_reason="MAX_TOKENS")
        == TRUNCATED_THINK_FEEDBACK
    )


def test_get_lean_code_accepts_single_final_fence_with_trailing_whitespace():
    raw = """<think>
done
</think>
```lean4
theorem my_theorem : True := by
  trivial
```

"""

    assert get_lean_code(raw) == "\n  trivial"


def test_get_lean_code_accepts_multiple_fences_if_last_is_final_answer():
    raw = """```lean4
theorem first : True := by
  trivial
```
```lean4
theorem second : True := by
  trivial
```
"""

    assert get_lean_code(raw) == "\n  trivial"


def test_get_subgoal_tactic_code_strips_leading_by():
    raw = """```lean4
theorem my_theorem (h : True) (Child Sibling : Prop) : True := by
  have h_child : Child := by
    exact h
  have h_sibling : Sibling := by admit
  trivial
```"""
    skeleton = "have h_child : Child := sorry\nhave h_sibling : Sibling := sorry\ntrivial"

    assert get_subgoal_tactic_code(raw, skeleton, 0) == "exact h"


def test_get_subgoal_tactic_code_accepts_old_sibling_admit_shape():
    raw = """```lean4
theorem my_theorem (h : True) (Child Sibling : Prop) : True := by
  have h_child : Child := by
    exact h
  have h_sibling : Sibling := admit
  trivial
```"""
    skeleton = "have h_child : Child := sorry\nhave h_sibling : Sibling := sorry\ntrivial"

    assert get_subgoal_tactic_code(raw, skeleton, 0) == "exact h"


def test_get_subgoal_tactic_code_preserves_aligned_lines_after_by_newline():
    raw = """```lean4
theorem my_theorem (h : True) (Child Sibling : Prop) : True := by
  have h_child : Child := by
    exact h
    exact h
  have h_sibling : Sibling := by admit
  trivial
```"""
    skeleton = "have h_child : Child := sorry\nhave h_sibling : Sibling := sorry\ntrivial"

    assert get_subgoal_tactic_code(raw, skeleton, 0) == "exact h\nexact h"


def test_get_subgoal_tactic_code_accepts_target_by_newline_after_skeleton_by_space():
    raw = """```lean4
theorem my_theorem (x y : ℤ) (h₀ : y ^ 2 + 3 * (x ^ 2 * y ^ 2) = 30 * x ^ 2 + 517) (h1 : y ^ 2 % 3 = 1) : (y ^ 2 - 10) * (3 * x ^ 2 + 1) = 487 := by
  have h_expand : (y ^ 2 - 10) * (3 * x ^ 2 + 1) = 3 * x ^ 2 * y ^ 2 + y ^ 2 - 30 * x ^ 2 - 10 := by admit
  have h_sub : 3 * x ^ 2 * y ^ 2 + y ^ 2 - 30 * x ^ 2 = 497 := by
    linarith [h₀]
  rw [h_expand, h_sub]
  norm_num
```"""
    skeleton = (
        "have h_expand : (y ^ 2 - 10) * (3 * x ^ 2 + 1) = 3 * x ^ 2 * y ^ 2 + y ^ 2 - 30 * x ^ 2 - 10 := by sorry\n"
        "have h_sub : 3 * x ^ 2 * y ^ 2 + y ^ 2 - 30 * x ^ 2 = 497 := by sorry\n"
        "rw [h_expand, h_sub]\n"
        "norm_num"
    )

    assert get_subgoal_tactic_code(raw, skeleton, 1) == "linarith [h₀]"


def test_get_subgoal_tactic_code_partial_scaffold_fallback_extracts_target_decl():
    raw = """```lean4
theorem my_theorem (A B : Prop) : True := by
  have h_a : A := by admit
  have h_b : B := by
    intro x
    exact fixed
  sorry
```"""
    skeleton = "have h_a : A := sorry\nhave h_b : B := sorry\ntrivial"

    assert get_subgoal_tactic_code(raw, skeleton, 1) == ""
    assert (
        get_subgoal_tactic_code(
            raw,
            skeleton,
            1,
            allow_partial_scaffold=True,
        )
        == "intro x\nexact fixed"
    )


def test_get_subgoal_skeleton_code_extracts_mini_skeleton_with_new_sorries():
    raw = """```lean4
theorem my_theorem (h : True) (Child Sibling : Prop) : True := by
  have h_child : Child := by
    have h_part : True := sorry
    exact h
  have h_sibling : Sibling := by admit
  trivial
```"""
    skeleton = "have h_child : Child := sorry\nhave h_sibling : Sibling := sorry\ntrivial"

    assert get_subgoal_skeleton_code(raw, skeleton, 0) == (
        "have h_part : True := sorry\n"
        "exact h"
    )


def test_validate_skeleton_replacement_rejects_admit_and_naked_sorry():
    bad = """have h_part : True := sorry
admit
sorry"""

    assert validate_skeleton_replacement(bad) == INVALID_SKELETON_FEEDBACK


def test_validate_skeleton_replacement_accepts_named_leaf_sorries_only():
    good = """have h_left : True := sorry
have h_right : True := by sorry
exact And.intro h_left h_right"""

    assert validate_skeleton_replacement(good) == ""
