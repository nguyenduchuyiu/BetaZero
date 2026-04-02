from betazero.core.nodes import ProofState


def build_prompt(state: ProofState, action_type: str) -> str:
    instruction = "Write a Lean 4 tactic." if action_type == "tactic" \
             else "Write a Lean 4 skeleton proof using 'sorry'."
    return (
        f"Context:\n{state.context}\n\n"
        f"Goal:\n{state.goal}\n\n"
        f"Instruction: {instruction}\n\nCode:\n"
    )
