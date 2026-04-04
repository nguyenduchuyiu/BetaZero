"""Assigns structural dependencies reward (r_dep) to skeleton actions."""

from betazero.env.lean_env import LeanEnv
from betazero.policy.output_parser import get_lean_code
from betazero.search.graph import ANDORGraph
from betazero.search.sorrifier.stitcher import ProofStitcher
from .calculator import RewardCalculator


class DependencyRewardAssigner:
    """Orchestrates stitching and kernel analysis to assign dependency rewards."""

    def __init__(self, lean: LeanEnv, reward: RewardCalculator):
        self.lean = lean
        self.reward = reward

    def assign(self, graph: ANDORGraph) -> None:
        """Bottom-up dependency reward assignment using Expr Trees."""
        for action, parent_state in graph.parent_items():
            if action.action_type != "skeleton":
                continue
            
            # 1. Collect child proofs
            child_proofs = [graph.extract_proof_code(child) for child in action.children]
            
            # 2. Stitch code
            skeleton_code = get_lean_code(action.content)
            stitched_code = ProofStitcher.stitch(skeleton_code, child_proofs)
            full_compilable_code = self.lean._build_cmd(parent_state, stitched_code)
            
            # 3. Analyze through Kernel Expr Tree
            dep_analysis = self.lean.analyze_dependencies(full_compilable_code)
            
            # 4. Map outputs to Calculator format and assign
            mapped_analysis = {
                "core": dep_analysis.get("core_solved", []) + dep_analysis.get("core_failed", []),
                "benign": dep_analysis.get("benign", []),
                "malignant": dep_analysis.get("malignant", [])
            }
            
            r_dep_score = self.reward.r_dep(mapped_analysis)
            
            # Fatal penalty for missing core subgoals
            if dep_analysis.get("core_failed"):
                r_dep_score = -1.0 
                
            graph.set_r_dep(action, r_dep_score)