#!/usr/bin/env python3
import json
import random
import re
import time
import urllib.request
from pathlib import Path

from betazero.policy.prompt import (
    _SKELETON_INSTRUCTION,
    _OUTPUT_FORMAT_INSTRUCTION,
    _TACTIC_INSTRUCTION,
    _USER_BASE_INSTRUCTION,
    _format_chatml,
)

# ============================================================================
# Cấu hình
# ============================================================================
ROOT = Path(__file__).resolve().parent
PROBLEMS_DIR = ROOT / "problems" / "miniF2F-Valid"

LIMIT = 244
SHUFFLE = False
SEED = 42

API_URL = "http://127.0.0.1:18787/chat"
ACTION = "skeleton"  # "skeleton" hoặc "tactic"
MODE = "Instant"

TIMEOUT_MS = 240_000
TURN_DELAY_MS = 2000
PAUSE_LOGIN_MS = 1000

REQUESTS_PER_PROBLEM = 3
MAX_RETRIES_PER_CALL = 3
RETRY_DELAY_S = 2.0

EXTRA_CRITICAL_RULES = (
    "\nCRITICAL RULES:\n"
    "1. FOCUSED REASONING: Your <think> process MUST be step-by-step and deeply logical. Break the problem into clear milestones.\n"
    "2. SHOW YOUR WORK: Explicitly calculate algebraic steps, but DO NOT over-explain trivial arithmetic.\n"
    "3. DIRECT PATH: Stick strictly to the most promising mathematical path. DO NOT simulate endless alternative approaches or backtrack unnecessarily.\n"
)


def split_header_theorem(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    th_i = next((i for i, ln in enumerate(lines) if re.match(r"^\s*theorem\b", ln)), None)
    if th_i is None:
        return "", text.strip()
    header = "\n".join(lines[:th_i]).strip()
    theorem = "\n".join(lines[th_i:]).strip()
    return header, theorem


def format_problem_block(lean: str) -> str:
    body = lean if lean.endswith("\n") else lean + "\n"
    return "[PROBLEM]\n```lean4\n" + body + "```"


def action_instruction() -> str:
    if ACTION == "tactic":
        return _TACTIC_INSTRUCTION
    if ACTION == "skeleton":
        return _SKELETON_INSTRUCTION
    raise ValueError(ACTION)


def build_prompt_from_lean(code_to_verify: str) -> str:
    full_system = action_instruction() + "\n\n" + _OUTPUT_FORMAT_INSTRUCTION
    user_msg = EXTRA_CRITICAL_RULES + _USER_BASE_INSTRUCTION + "\n\n" + format_problem_block(code_to_verify)
    return _format_chatml(full_system, user_msg)


def get_deepseek_response(prompt: str) -> str | None:
    payload = {
        "mode": MODE,
        "pauseForLoginMs": PAUSE_LOGIN_MS,
        "turnDelayMs": TURN_DELAY_MS,
        "timeoutMs": TIMEOUT_MS,
        "prompts": [prompt],
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=(TIMEOUT_MS // 1000) + 60) as resp:
            data = json.loads(resp.read().decode())
            out = data.get("result", {}).get("turns", [{}])[-1].get("response")
            if out is not None and not str(out).strip():
                return None
            return out
    except Exception as e:
        print(f"Lỗi gọi API: {e}")
        return None


def main() -> None:
    paths = sorted(PROBLEMS_DIR.glob("*.lean"))
    if not paths:
        print(f"Không thấy file .lean trong {PROBLEMS_DIR}")
        return

    indices = list(range(len(paths)))
    if SHUFFLE:
        random.Random(SEED).shuffle(indices)

    n_sent = 0
    for idx in indices:
        if n_sent >= LIMIT:
            break
        p = paths[idx]
        raw = p.read_text(encoding="utf-8")
        header, theorem = split_header_theorem(raw)
        code_to_verify = f"{header}\n{theorem}".strip() if header else theorem

        prompt = build_prompt_from_lean(code_to_verify)
        for turn in range(1, REQUESTS_PER_PROBLEM + 1):
            resp: str | None = None
            for attempt in range(1, MAX_RETRIES_PER_CALL + 1):
                resp = get_deepseek_response(prompt)
                if resp is not None:
                    break
                print(
                    f"{p.name} req {turn}/{REQUESTS_PER_PROBLEM}: "
                    f"lỗi/rỗng ({attempt}/{MAX_RETRIES_PER_CALL}), chờ {RETRY_DELAY_S}s..."
                )
                if attempt < MAX_RETRIES_PER_CALL:
                    time.sleep(RETRY_DELAY_S)
            if resp is None:
                print(f"{p.name}: dừng sau {MAX_RETRIES_PER_CALL} lần ở request {turn}/{REQUESTS_PER_PROBLEM}.")
                break
            print(f"{p.name} req {turn}/{REQUESTS_PER_PROBLEM} ok")
        else:
            n_sent += 1
            print(f"[{n_sent}/{LIMIT}] {p.name} xong {REQUESTS_PER_PROBLEM} request")

    print(f"Hoàn thành. Đủ 3/3 request cho {n_sent} bài (giới hạn LIMIT={LIMIT}).")


if __name__ == "__main__":
    main()
