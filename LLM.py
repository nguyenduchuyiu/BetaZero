from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread

from betazero.core.nodes import ProofState
from betazero.policy.output_parser import get_lean_code
from betazero.policy.prompt import build_prompt


MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # override via env/replace if needed
ACTION_TYPE = "tactic"  # "tactic" | "skeleton"
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.7
TOP_P = 0.95
SEED = 42

CONTEXT = """\
x b : ℝ
h₀ : 0 < b
h₁ : (7 : ℝ) ^ (x + 7) = 8 ^ x
h₂ : x = Real.logb b (7 ^ 7)
"""
GOAL = "b = 8 / 7"


def main() -> int:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=(torch.bfloat16 if dev == "cuda" else None),
        device_map=("auto" if dev == "cuda" else None),
        trust_remote_code=True,
    )

    state = ProofState(context=CONTEXT.strip(), goal=GOAL.strip(), header="")
    prompt = build_prompt(state, ACTION_TYPE)

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    print("=" * 80)
    print("PROMPT:")
    print(prompt)
    print("=" * 80)
    print("RAW OUTPUT (stream):")

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=TEMPERATURE > 0,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        streamer=streamer,
    )

    t = Thread(target=lambda: model.generate(**gen_kwargs), daemon=True)
    t.start()

    chunks: list[str] = []
    for piece in streamer:
        print(piece, end="", flush=True)
        chunks.append(piece)
    t.join()

    gen = "".join(chunks)
    extracted = get_lean_code(gen)
    print("\n")
    print("=" * 80)
    print("EXTRACTED LEAN CODE (proof body after := by / by):")
    print(extracted if extracted else "<EMPTY>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

