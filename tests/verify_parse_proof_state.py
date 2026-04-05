#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify skeleton Lean trên REPL thật, rồi trích goal qua LeanEnv._parse_proof_state.

  cd /path/to/BetaZero && python tests/verify_parse_proof_state.py

REPL đã warmup `import Mathlib` vào base_env — không được thêm `import` trong từng `cmd`
(Lean báo: invalid 'import' command, it must be used in the beginning of the file).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


_SKELETONS: list[tuple[str, str]] = [
    (
        "1. test_sorry",
        """example (a b : Nat) : a = b → b = a := by
  intro h
  sorry
""",
    ),
    (
        "2. test_induction (dừng sau induction)",
        """theorem test_induction (n : ℕ) : n + 0 = n := by
  induction n
  -- Dừng ở đây và xem output của REPL""",
    ),
    (
        "3. test_calc",
        """theorem test_calc (a b c : ℕ) (h1 : a = b) (h2 : b = c) : a = c := by
  calc
    a = b := by exact h1
    _ = c := by sorry""",
    ),
    (
        "4. test_extra (rfl rồi have sorry)",
        """theorem test_extra (n : ℕ) : n = n := by
  rfl
  have h : 1 = 1 := by sorry""",
    ),
]


def _short_err(errors: list) -> str:
    if not errors:
        return ""
    e0 = errors[0]
    return (e0.get("data") or str(e0))[:500]


def main() -> None:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)

    from betazero.env.lean_env import LeanEnv
    from betazero.env.lean_verifier import Lean4ServerScheduler

    sch = Lean4ServerScheduler(max_concurrent_requests=1, timeout=120, name="verify_parse")
    try:
        for name, body in _SKELETONS:
            code = body
            vr = sch.verify(code)
            sep = "=" * 72
            print(sep)
            print(name)
            print("--- Lean ---")
            print(body)
            print("--- verify() ---")
            print(f"  pass={vr.get('pass')}  complete={vr.get('complete')}")
            if vr.get("errors"):
                print(f"  errors[0]: {_short_err(vr['errors'])}")
            if vr.get("warnings"):
                print(f"  n_warnings={len(vr['warnings'])}")
            sorries = vr.get("sorries") or []
            print(f"  n_sorries={len(sorries)}")
            for i, s in enumerate(sorries):
                raw = s.get("goal", "") or ""
                ps = LeanEnv._parse_proof_state(raw, header="")
                print(f"  --- sorry[{i}] raw goal (repr, trim 2000) ---")
                r = repr(raw)
                print(f"  {r[:2000]}{'…' if len(r) > 2000 else ''}")
                print("  --- _parse_proof_state ---")
                print(f"  context: {ps.context!r}")
                print(f"  goal:    {ps.goal!r}")
            print()
    finally:
        sch.close()
    print("Done.")


if __name__ == "__main__":
    main()
