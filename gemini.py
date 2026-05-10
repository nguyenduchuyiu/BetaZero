import os
import sys
from dotenv import load_dotenv

# Add workspace to path
sys.path.append("/workspace/npthai/BetaZero")

from betazero.policy.gemini_server import GeminiAPIServer
from betazero.utils.config import Config
from betazero.core import ProofState

def test_gemini():
    load_dotenv()
    cfg = Config()
    cfg.max_new_tokens = 8192
    cfg.temperature = 1.0
    
    server = GeminiAPIServer(cfg)
    server.start()
    
    # Mock ProofState
    state = ProofState(
        header="import Mathlib",
        context="(a b c : ℝ)(h₀ : 0 < a ∧ 0 < b ∧ 0 < c)(h₁ : 3 ≤ a * b + b * c + c * a)",
        goal=" 3 / Real.sqrt 2 ≤ a / Real.sqrt (a + b) + b / Real.sqrt (b + c) + c / Real.sqrt (c + a)"
    )
    
    print("\n--- Testing Sample with Mock ProofState ---")
    from betazero.policy.prompt import build_prompt
    prompt = build_prompt(state, "tactic")
    print(f"Built Prompt:\n{prompt}\n")
    
    results = server.sample([state], action_type="tactic", n=1)
    
    for i, res_list in enumerate(results):
        print(f"State {i} results:")
        for j, res in enumerate(res_list):
            print(f"  Completion {j}:")
            print(res["text"])

if __name__ == "__main__":
    test_gemini()
