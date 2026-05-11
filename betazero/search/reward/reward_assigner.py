"""Assigns structural dependencies reward (r_dep) to skeleton actions."""

from betazero.env.lean_env import LeanEnv
from betazero.policy.output_parser import get_lean_code
from betazero.search.graph import ANDORGraph
from betazero.search.sorrifier.stitcher import ProofStitcher
from betazero.utils.lean_cmd import build_theorem
from .calculator import RewardCalculator
import re


class DependencyRewardAssigner:
    """Orchestrates stitching and kernel analysis to assign dependency rewards."""

    def __init__(self, lean: LeanEnv, reward: RewardCalculator):
        self.lean = lean
        self.reward = reward

    def _extract_sorry_vars(self, code: str) -> set[str]:
        """Parses Lean code indentation to accurately attribute 'sorry' to its declaring variable."""
        lines = code.splitlines()
        sorry_vars = set()
        stack = []
        
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                continue
            indent = len(line) - len(stripped)
            
            while stack and indent <= stack[-1][0]:
                stack.pop()
                
            match = re.match(r"(?:have|let)\s+([a-zA-Z0-9_]+)\s*[:=]", stripped)
            if match:
                var_name = match.group(1)
                stack.append((indent, var_name))
                
            if re.search(r"\bsorry\b", stripped):
                if stack:
                    sorry_vars.add(stack[-1][1])
                    
        return sorry_vars

    def assign(self, graph: ANDORGraph) -> None:
        """Bottom-up dependency reward assignment using Expr Trees."""
        for action, parent_state in graph.parent_items():
            if action.action_type != "skeleton":
                continue

            # Chặn đứng! Bỏ qua các xác chết (0 con) do FailureHandler vứt lại
            if not action.children:
                continue
            
            # 1. Collect child proofs
            child_proofs = [graph.extract_proof_code(child) for child in action.children]
            
            # 2. Stitch code
            stitched_code = ProofStitcher.stitch(action.extracted_code, child_proofs)
            full_compilable_code = build_theorem(parent_state, stitched_code)
            
            # 3. Analyze through Kernel Expr Tree
            # Extract allowed subgoals to prevent 'farming core variables'
            # Uses indentation-aware parser to correctly handle nested sorries (e.g., inside `calc`)
            allowed_vars = self._extract_sorry_vars(action.extracted_code)
            
            dep_analysis = self.lean.analyze_dependencies(full_compilable_code, allowed_vars=allowed_vars)
            
            # 4. Map outputs to Calculator format and assign
            if len(dep_analysis.get("core_failed", [])) > 0:
                r_dep_score = 0.0
            else:
                mapped_analysis = {
                    "core": dep_analysis.get("core_solved", []),
                    "benign": dep_analysis.get("benign", []),
                    "malignant": dep_analysis.get("malignant", [])
                }
                r_dep_score = self.reward.r_dep(mapped_analysis)
                
                # Nếu không có core_failed, chứng tỏ các subgoal chứa sorry chỉ là malignant/benign.
                # Do đó skeleton này THỰC CHẤT đã được solved!
                # Guard: chỉ set override khi action có code thực sự VÀ r_env đạt 1.0 (không có lỗi cú pháp/logic)
                if action.extracted_code and graph.get_r_env(action) == 1.0:
                    # Garbage Collection: Dọn dẹp các biến rác (malignant/benign)
                    garbage_vars = dep_analysis.get("malignant", []) + dep_analysis.get("benign", [])
                    if garbage_vars:
                        # GHI VÀO GRAPH thay vì ghi vào action
                        graph.set_garbage_vars(action, garbage_vars)
                    
                    graph.set_skeleton_override(action, True)
            
            graph.set_r_dep(action, r_dep_score)