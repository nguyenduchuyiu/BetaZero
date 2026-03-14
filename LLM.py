import argparse
import os
import sys
import textwrap
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer


# MODEL_ID = "AI-MO/Kimina-Prover-Preview-Distill-7B"
MODEL_ID = "Goedel-LM/Goedel-Prover-V2-8B"


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
        device_map="cuda",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str, 
        default=textwrap.dedent(
            """
            <|im_start|>system
            You are an expert in mathematics and Lean 4.
            Given a Lean proof state, only produce Lean 4 Proof with `have` Statements,
            sketch a skeleton of solution in Lean 4 
            (ONLY SKELETON with sorry placeholders), which are hypothesis, 
            subgoals that needed to prove the main statement, not detail tatics.
            Dont output Complete Lean 4 Proof.
            Example output:
            ```lean4
            theorem example := by
                have h1:= by sorry
                have h2:= by sorry
                have h3:= by sorry
                exact h3
            ```
            <|im_end|>

            <|im_start|>user
            ```lean4
            import Mathlib
            open BigOperators Real Nat Topology Rat

            theorem aime_1983_p1 (x y z w : ℕ) (ht : 1 < x ∧ 1 < y ∧ 1 < z) (hw : 0 ≤ w)
                (h0 : Real.log w / Real.log x = 24) (h1 : Real.log w / Real.log y = 40)
                (h2 : Real.log w / Real.log (x * y * z) = 12) : Real.log w / Real.log z = 60 := by
                    sorry
            ```
            <|im_end|>

            <|im_start|>assistant
            ### Informal Statement

            Let $x, y, z$, and $w$ be natural numbers such that $x > 1$, $y > 1$, $z > 1$, and $w \ge 0$. Given the following logarithmic equations:

            * $\frac{\ln w}{\ln x} = 24$
            * $\frac{\ln w}{\ln y} = 40$
            * $\frac{\ln w}{\ln(xyz)} = 12$

            Prove that $\frac{\ln w}{\ln z} = 60$.

            *(Note: This is equivalent to saying if $\log_x w = 24$, $\log_y w = 40$, and $\log_{xyz} w = 12$, prove that $\log_z w = 60$.)*

            ---

            ### Solution Skeleton

            **Hypotheses**

            * **Domain:** $x, y, z, w \in \mathbb{N}$
            * **Base Constraints:** $x > 1$, $y > 1$, $z > 1$ (This ensures $\ln x > 0$, $\ln y > 0$, $\ln z > 0$)
            * **Argument Constraint:** $w \ge 0$
            * **$H_1$:** $\frac{\ln w}{\ln x} = 24$ (which rewrites to $\ln x = \frac{\ln w}{24}$)
            * **$H_2$:** $\frac{\ln w}{\ln y} = 40$ (which rewrites to $\ln y = \frac{\ln w}{40}$)
            * **$H_3$:** $\frac{\ln w}{\ln(xyz)} = 12$ (which rewrites to $\ln(xyz) = \frac{\ln w}{12}$)

            **Subgoals / Logical Steps**

            1. **Expand the Product Logarithm:** Use logarithmic properties to break down the denominator of the third hypothesis: prove that $\ln(xyz) = \ln x + \ln y + \ln z$.
            2. **Algebraic Substitution:** Substitute the re-written forms of $H_1$, $H_2$, and the expanded product from Step 1 into $H_3$ to form an equation linking $\ln z$ directly to $\ln w$.
            > **Target Equation:** $\frac{\ln w}{12} = \frac{\ln w}{24} + \frac{\ln w}{40} + \ln z$


            3. **Isolate $\ln z$:** Rearrange the substituted equation to solve for $\ln z$ in terms of $\ln w$.
            > **Target Equation:** $\ln z = \ln w \cdot \left(\frac{1}{12} - \frac{1}{24} - \frac{1}{40}\right)$


            4. **Arithmetic Simplification:** Prove the arithmetic reduction of the fractions: $\frac{1}{12} - \frac{1}{24} - \frac{1}{40} = \frac{1}{60}$.
            5. **Final Rearrangement:** Apply the result from Step 4 to show $\ln z = \frac{\ln w}{60}$, and algebraically manipulate this to reach the final goal: $\frac{\ln w}{\ln z} = 60$.
        """
        ).strip(),
        help="Text prompt to send to the model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    print("\n=== Model reply (streaming) ===\n")
    generate_reply(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print()  # newline at end


if __name__ == "__main__":
    main()
