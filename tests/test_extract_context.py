#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

from betazero.env.lean_verifier import Lean4ServerScheduler
from betazero.utils.lean_parse import parse_proof_state

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SOLVE_B = """
open Nat Real
theorem my_theorem (x b : ℝ)
  (h₀ : 0 < b)
  (h₁ : (7 : ℝ) ^ (x + 7) = 8 ^ x)
  (h₂ : x = Real.logb b (7 ^ 7))
  : b = 8 / 7 := by
  have h3 : (7 : ℝ) ^ x * 7 ^ 7 = 8 ^ x := sorry
  have h4 : (7 : ℝ) ^ 7 = (8 / 7) ^ x := sorry
  have h5 : b ^ x = 7 ^ 7 := sorry
  have h6 : b ^ x = (8 / 7) ^ x := sorry
  have h7 : x ≠ 0 := sorry
  exact sorry
"""


def main() -> int:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)

    sch = Lean4ServerScheduler(max_concurrent_requests=1, timeout=120, name="first_sorry_real")
    try:
        vr = sch.verify(SOLVE_B.strip())
        print("pass=", vr.get("pass"), "complete=", vr.get("complete"))
        if vr.get("errors"):
            e0 = vr["errors"][0]
            print("error[0]:", (e0.get("data") or str(e0))[:800])
        sorries = vr.get("sorries") or []
        print("n_sorries=", len(sorries))
        if not sorries:
            return 1
        # Thứ tự sorries do Lean quyết định; sorry "đầu tiên trong file" có thể là index khác 0.
        for i, s in enumerate(sorries):
            raw = s.get("goal", "") or ""
            ps = parse_proof_state(raw, header="open Nat Real")
            print(f"--- sorry[{i}] (parsed) ---")
            print("context:\n", ps.context)
            print("goal:\n", ps.goal)
        return 0
    finally:
        sch.close()


if __name__ == "__main__":
    raise SystemExit(main())
