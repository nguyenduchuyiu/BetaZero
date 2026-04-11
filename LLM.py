from __future__ import annotations

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread

from betazero.core.nodes import ProofState
from betazero.policy.output_parser import get_lean_code
from betazero.policy.prompt import build_prompt


MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # override via env/replace if needed
ACTION_TYPE = "skeleton"  # "tactic" | "skeleton"
_DEFAULT_LORA = "lora_skeleton_model"
LORA_PATH = os.environ.get("LORA_PATH", _DEFAULT_LORA).strip()
if LORA_PATH and not os.path.isdir(LORA_PATH):
    LORA_PATH = ""
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.7
TOP_P = 0.95
SEED = 42

CONTEXT = """\
x : ℝ
h₀ : 0 < x
h₁ : Real.logb 2 (Real.logb 8 x) = Real.logb 8 (Real.logb 2 x)
"""
GOAL = "Real.logb 2 x ^ 2 = 27"


def main() -> int:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=dtype if dev == "cuda" else None,
        device_map=("auto" if dev == "cuda" else None),
        trust_remote_code=True,
    )
    if LORA_PATH:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, LORA_PATH, is_trainable=False)

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

