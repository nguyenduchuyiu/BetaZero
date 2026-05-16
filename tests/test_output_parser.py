import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gammazero.policy.output_parser import get_lean_code


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
