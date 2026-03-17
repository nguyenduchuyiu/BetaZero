import argparse
import os
import sys
import textwrap
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer


# MODEL_ID = "AI-MO/Kimina-Prover-Preview-Distill-7B"
# MODEL_ID = "Goedel-LM/Goedel-Prover-V2-8B"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def load_model(model_id: str = MODEL_ID):
    """
    Load tokenizer/model with automatic GPU+CPU sharding (needs accelerate).
    If accelerate is missing, instruct how to install.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="cpu",
    )

    return tokenizer, model


def generate_reply(prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    tokenizer, model = load_model()

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    # Launch generation in a background thread
    import threading

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # Stream tokens to stdout
    output_chunks = []
    for token in streamer:
        sys.stdout.write(token)
        sys.stdout.flush()
        output_chunks.append(token)

    thread.join()
    return "".join(output_chunks)


with open("nl_h2_output.txt", "r") as f:
    PLANNER_OUTPUT = f.read()

PLANNER_OUTPUT = PLANNER_OUTPUT.replace("<think>", "").replace("</think>", "")

FL_PROMPT = f"""
<|im_start|>system
You are a Lean 4 Skeleton Generator. Your ONLY task is to output 'have' statements.
**STRICT RULES:**
1. NO PROSE. NO EXPLANATIONS. NO CALCULATIONS.
2. Every line must follow the format: `have hX : type := by sorry`
3. If you output anything other than Lean code, you fail.
<|im_end|>
<|im_start|>user
Decompose the following theorem into a set of necessary intermediate lemmas/subgoals:
Hypotheses:
    a b : ℝ
    h₀ : Real.logb 8 a + Real.logb 4 (b ^ 2) = 5
    h₁ : Real.logb 8 b + Real.logb 4 (a ^ 2) = 7
Main Goal:
    Real.logb 2 a = 6
<|im_end|>
<|im_start|>assistant
<think>
{PLANNER_OUTPUT}
I need to formalize this in Lean 4 to the `have` statements with `sorry` like the following format:
```lean4
have h1 : type := by sorry
have h2 : type := by sorry
...
have hn : type := by sorry
```
</think>
```lean4
theorem 
"""

ATOMIC_PROMPT = """
<|im_start|>system
You are a Tactical Logic Decomposer. 
Your task is to take a SPECIFIC complex subgoal and break it down into smaller, atomic verifiable steps.

**INSTRUCTION:**
1. Decompose ONLY this specific goal.
2. Provide a sequence of `have` statements that bridge the current state to this subgoal.
3. Each step should be "atomic" enough for a basic tactic to handle.
4. Use a nested structure:
   `have parent_goal : type := by`
   `  have sub_step1 : type := by sorry`
   ...
<|im_end|>

<|im_start|>user

The parent goal is: #TODO: add parent goal
The current solver (linarith/aesop) cannot close this goal in one step.
<|im_end|>

<|im_start|>assistant
<think>
"""

NL_PROMPT = """
<|im_start|>system
You are a Formal Logic Architect for Lean 4. Your task is to decompose a theorem into a minimal sequence of high-level verifiable milestones (sub-lemmas).

**ARCHITECTURAL STRATEGY:**
1. **Master Strategist Abstraction:** Do not document calculations. Document "States of Truth". 
2. **The "Tactic-Power" Assumption:** Assume `linarith`, `aesop`, `polyrith`, and `norm_num` are available. If a transition can be solved by these tactics in one go, DO NOT create a subgoal for it.
3. **Domain Bottlenecks:** Create subgoals only when shifting mathematical domains (e.g., Logarithmic Constraints -> Linear System -> Numerical Solution).
4. **Milestone Limit:** Aim for 3-6 `have` statements. Quality over quantity.

**EXAMPLE:**
- BAD (Atomic): 
  `have h1 : Real.logb 8 a = Real.log a / Real.log 8 := by sorry`
- GOOD (Milestone): 
  `have h_linear : (1/3 : ℝ) * Real.logb 2 a + Real.logb 2 b = 5 := by sorry`

**OUTPUT FORMAT:**
- Output ONLY the Lean 4 code block.
- Start directly with the `theorem` statement or the sequence of `have` statements.
- Every subgoal MUST end with `:= by sorry`.
Example:
```lean4
theorem log_b (a b : ℝ) 
  (h₀ : Real.logb 8 a + Real.logb 4 (b ^ 2) = 5)
  (h₁ : Real.logb 8 b + Real.logb 4 (a ^ 2) = 7)
  : a * b = 512 := by
  have h2 : Real.logb 2 a = 6 := by sorry
  have h3 : Real.logb 2 b = 3 := by sorry
  have h4 : a = 64 := by sorry
  have h5 : b = 8 := by sorry
<|im_end|>

<|im_start|>user
Decompose the following theorem into a set of necessary intermediate lemmas/subgoals:
Hypotheses:
    a b : ℝ
    h₀ : Real.logb 8 a + Real.logb 4 (b ^ 2) = 5
    h₁ : Real.logb 8 b + Real.logb 4 (a ^ 2) = 7
Main Goal:
    a * b = 512
<|im_end|>

<|im_start|>assistant
<think>
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str, 
        default=NL_PROMPT,
        help="Text prompt to send to the model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    generate_reply(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print()  # newline at end


if __name__ == "__main__":
    main()
