from betazero.core.nodes import ProofState
from betazero.utils.lean_cmd import build_theorem
from betazero.policy.prompt import build_prompt
state = ProofState("contetx", "goal")
print(build_prompt(state, "skeleton"))