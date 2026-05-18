import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gammazero.policy.output_parser import (
    get_lean_code,
    get_subgoal_skeleton_code,
    get_subgoal_tactic_code,
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
