from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick Gemini API smoke test.")
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        help="Gemini model name. Default: GEMINI_MODEL or gemini-3-flash-preview.",
    )
    parser.add_argument(
        "--prompt",
        default="Say hello in Vietnamese and solve 2 + 2 in one short sentence.",
        help="Prompt to send.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="0 disables thinking for supported models; -1 lets Gemini choose dynamically.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Missing GOOGLE_API_KEY or GEMINI_API_KEY in environment/.env")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=args.model,
        contents=args.prompt,
        config=types.GenerateContentConfig(
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=args.thinking_budget),
        ),
    )

    print(f"model: {args.model}")
    if response.candidates:
        print(f"finish_reason: {response.candidates[0].finish_reason}")
    if response.usage_metadata:
        print(f"usage: {response.usage_metadata}")
    print("\n--- response ---")
    print(response.text or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
