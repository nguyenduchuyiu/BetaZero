#!/usr/bin/env python3
import sys
import json
import random
import re
import time
import urllib.request
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Thêm thư mục gốc của project vào sys.path để import được betazero
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from betazero.core.nodes import ProofState
from betazero.policy.prompt import build_prompt, build_messages
from betazero.policy.output_parser import get_lean_code

# ============================================================================
# Cấu hình
# ============================================================================
PROBLEMS_DIR = ROOT / "problems" / "miniF2F-Valid"
SUBGOALS_FILE = ROOT / "data" / "unsolved_subgoals.jsonl"
OUTPUT_FILE = ROOT / "data" / "tactic_samples.jsonl"

LIMIT = 3354
SHUFFLE = False
SEED = 42

# Proxy Configuration (Lưu ý: API_URL này dành cho proxy cục bộ hoặc custom server)
API_URL = "http://127.0.0.1:8787/chat"
MODE = "Instant"

# Official DeepSeek API Configuration (Beta Prefix Completion)
USE_API = True  # Chuyển sang True để dùng API trực tiếp từ deepseek.com
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Thay bằng API key của bạn hoặc set env var
DEEPSEEK_BASE_URL = "https://api.deepseek.com/beta"
DEEPSEEK_MODEL = "deepseek-chat"
STOP_SEQUENCES = None # Ví dụ: ["```"] nếu dùng prefix code block
BATCH_SIZE = 10     # Số lượng request API chạy song song
SAVE_LOCK = threading.Lock()

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
    theorem_raw = "\n".join(lines[th_i:]).strip()
    theorem = re.sub(r"(\s*theorem\s+)\w+", r"\1my_theorem", theorem_raw, count=1)
    return header, theorem


def build_prompt_from_lean(code_to_verify: str, action_type: str = "skeleton") -> str:
    header, theorem = split_header_theorem(code_to_verify)
    ps = ProofState(
        context="",
        goal=theorem.replace(":= by sorry", "").strip(),
        header=header
    )
    return build_prompt(ps, action_type=action_type, extra_rules=EXTRA_CRITICAL_RULES)


def build_tactic_prompt_for_subgoal(context: str, goal: str, header: str = "") -> str:
    ps = ProofState(
        context=context,
        goal=goal,
        header=header if header else "import Mathlib\nopen BigOperators Real Nat Topology Rat"
    )
    return build_prompt(ps, action_type="tactic", extra_rules=EXTRA_CRITICAL_RULES)


def build_messages_from_lean(code_to_verify: str, action_type: str = "skeleton") -> list[dict]:
    header, theorem = split_header_theorem(code_to_verify)
    ps = ProofState(
        context="",
        goal=theorem.replace(":= by sorry", "").strip(),
        header=header
    )
    return build_messages(ps, action_type=action_type, extra_rules=EXTRA_CRITICAL_RULES)


def build_tactic_messages_for_subgoal(context: str, goal: str, header: str = "") -> list[dict]:
    ps = ProofState(
        context=context,
        goal=goal,
        header=header if header else "import Mathlib\nopen BigOperators Real Nat Topology Rat"
    )
    return build_messages(ps, action_type="tactic", extra_rules=EXTRA_CRITICAL_RULES)


def get_deepseek_response(prompt_or_messages: str | list[dict]) -> str | None:
    if USE_API:
        return _get_official_api_response(prompt_or_messages)
    
    # Proxy logic cũ
    payload = {
        "mode": MODE,
        "pauseForLoginMs": PAUSE_LOGIN_MS,
        "turnDelayMs": TURN_DELAY_MS,
        "timeoutMs": TIMEOUT_MS,
        "prompts": [prompt_or_messages] if isinstance(prompt_or_messages, str) else [str(prompt_or_messages)],
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
        print(f"Lỗi gọi API Proxy: {e}")
        return None


def _get_official_api_response(messages: list[dict]) -> str | None:
    if not isinstance(messages, list):
        print("Lỗi: Official API yêu cầu danh sách messages (role/content).")
        return None
    
    # Chuẩn bị messages cho Prefix Completion (Beta)
    # Theo hướng dẫn, tin nhắn cuối phải là role assistant và có prefix=True
    api_messages = [msg.copy() for msg in messages]
    if api_messages and api_messages[-1].get("role") == "assistant":
        api_messages[-1]["prefix"] = True
    
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": api_messages,
        "max_tokens": 4096
    }
    if STOP_SEQUENCES:
        payload["stop"] = STOP_SEQUENCES

    req = urllib.request.Request(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            content = data["choices"][0]["message"]["content"]
            
            # Ghép prefix gốc với phần content trả về để có chuỗi hoàn chỉnh
            prefix_val = api_messages[-1].get("content", "")
            return prefix_val + content
    except Exception as e:
        print(f"Lỗi gọi Official DeepSeek API: {e}")
        return None


def save_api_response(save_data: dict) -> None:
    """Lưu phản hồi từ API vào file JSONL để tối ưu hiệu năng ghi (thread-safe)."""
    if not USE_API:
        return
    
    with SAVE_LOCK:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(save_data, ensure_ascii=False) + "\n")


def sample_skeletons() -> None:
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

        if USE_API:
            prompt_data = build_messages_from_lean(code_to_verify, action_type="skeleton")
        else:
            prompt_data = build_prompt_from_lean(code_to_verify, action_type="skeleton")

        def _one_request(turn):
            resp: str | None = None
            for attempt in range(1, MAX_RETRIES_PER_CALL + 1):
                resp = get_deepseek_response(prompt_data)
                if resp is not None:
                    break
                print(f"{p.name} req {turn}: lỗi ({attempt}/{MAX_RETRIES_PER_CALL})")
                if attempt < MAX_RETRIES_PER_CALL:
                    time.sleep(RETRY_DELAY_S)
            
            if resp:
                from betazero.policy.prompt import build_prompt
                ps = ProofState(context="", goal=theorem.replace(":= by sorry", "").strip(), header=header)
                full_prompt_str = build_prompt(ps, action_type="skeleton", extra_rules=EXTRA_CRITICAL_RULES)
                
                print(f"{p.name} req {turn} ok")
                return {
                    "id": f"{p.stem}_{turn}",
                    "theorem": code_to_verify,
                    "prompt": full_prompt_str,
                    "raw_output": resp,
                    "extracted_code": get_lean_code(resp)
                }
            return None

        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            results = list(executor.map(_one_request, range(1, REQUESTS_PER_PROBLEM + 1)))
        
        for res in results:
            if res:
                save_api_response(res)
        
        n_sent += 1
        print(f"[{n_sent}/{LIMIT}] {p.name} xong")


def sample_tactics():
    """Chỉ đọc subgoals và gửi request, không lưu payload json (tải data thủ công qua conversations.json)."""
    if not SUBGOALS_FILE.exists():
        print(f"❌ Không tìm thấy file {SUBGOALS_FILE}")
        return

    with open(SUBGOALS_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    n_sent = 0
    for item in data:
        if n_sent >= LIMIT:
            break
            
        thm_id = item.get("id", "Unknown")
        subgoals = item.get("subgoals", [])
        if not subgoals:
            continue

        print(f"--- Bắt đầu gửi tactic cho {thm_id} ({len(subgoals)} subgoals) ---")
        
        def _process_sg(args):
            j, sg = args
            context = sg.get("context", "")
            goal = sg.get("goal", "")
            sg_id = f"{thm_id}_sg_{j}"
            
            if USE_API:
                msg_input = build_tactic_messages_for_subgoal(context, goal, header=item.get("header", ""))
            else:
                msg_input = build_tactic_prompt_for_subgoal(context, goal, header=item.get("header", ""))
            
            resp = None
            for attempt in range(MAX_RETRIES_PER_CALL):
                resp = get_deepseek_response(msg_input)
                if resp: break
                time.sleep(RETRY_DELAY_S)
            
            if resp: 
                from betazero.policy.prompt import build_prompt
                from betazero.utils.lean_cmd import build_theorem
                ps = ProofState(context=context, goal=goal, header=item.get("header", "import Mathlib\nopen BigOperators Real Nat Topology Rat"))
                full_prompt_str = build_prompt(ps, action_type="tactic", extra_rules=EXTRA_CRITICAL_RULES)
                
                print(f"  [{sg_id}] OK")
                return {
                    "id": sg_id,
                    "theorem": build_theorem(ps, "sorry", name="my_theorem"),
                    "prompt": full_prompt_str,
                    "raw_output": resp,
                    "extracted_code": get_lean_code(resp)
                }
            else:
                print(f"  [{sg_id}] FAIL")
                return None

        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            results = list(executor.map(_process_sg, enumerate(subgoals)))
        
        for res in results:
            if res:
                save_api_response(res)

        n_sent += 1

    print("✅ Hoàn tất gửi yêu cầu!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "tactic":
        sample_tactics()
    else:
        sample_skeletons()
