"""
Lean 4 Expression Tree Dependency Analyzer.
-------------------------------------------
Uses De Bruijn index tracking to deeply analyze variable usage and sorry 
occurrences within Lean 4 elaborator expression trees (AST/Expr JSON).
"""

from __future__ import annotations
from typing import Dict, List, Any


class ExprDependencyAnalyzer:
    """
    Analyzes Lean 4 Expr JSON trees to classify subgoals (letE) and 
    detect unused context parameters (forallE/lam) via De Bruijn tracking.
    """

    def _contains_sorry(self, node: Any) -> bool:
        """Recursively checks if a node or its children contain the `sorryAx` constant."""
        if not isinstance(node, dict):
            return False
        
        # Base case: Found the sorry axiom
        if node.get("expr") == "const" and node.get("name") == "sorryAx":
            return True
            
        # Recursive case
        for val in node.values():
            if isinstance(val, dict) and self._contains_sorry(val):
                return True
        return False

    def _is_bvar_used(self, node: Any, target_idx: int) -> bool:
        """
        Recursively checks if a De Bruijn variable (`bvar`) with `target_idx` 
        is used in the given expression tree. Safely shifts indices across binders.
        """
        if not isinstance(node, dict):
            return False

        expr_type = node.get("expr")

        # 1. Base case: Found a bound variable
        if expr_type == "bvar":
            return node.get("idx") == target_idx

        # 2. Binder case: `lam` (lambda) and `forallE` (Pi type)
        # These introduce exactly ONE new variable in their `body`.
        if expr_type in ("lam", "forallE"):
            return (
                self._is_bvar_used(node.get("var_type", {}), target_idx) or
                # Shift index by +1 when entering the binder's body
                self._is_bvar_used(node.get("body", {}), target_idx + 1)
            )

        # 3. Binder case: `letE` (Let binding)
        # A `let` introduces a variable in its `body`, but NOT in its type or value.
        elif expr_type == "letE":
            return (
                self._is_bvar_used(node.get("var_type", {}), target_idx) or
                self._is_bvar_used(node.get("val", {}), target_idx) or
                # Shift index by +1 only for the body
                self._is_bvar_used(node.get("body", {}), target_idx + 1)
            )

        # 4. Standard case: Traverse all other nodes without shifting the index
        else:
            for val in node.values():
                if isinstance(val, dict) and self._is_bvar_used(val, target_idx):
                    return True
            return False

    def classify_skeleton_subgoals(self, root_expr: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Scans the Expr tree for `letE` nodes (which represent `have` tactics/subgoals).
        Classifies them into: core_solved, core_failed, malignant, benign.
        """
        results = {
            "core_solved": [],
            "core_failed": [],
            "malignant": [],
            "benign": []
        }

        def traverse(node: Any):
            if not isinstance(node, dict):
                return
            
            if node.get("expr") == "letE":
                var_name = node.get("var_name", "unknown")
                
                # We usually ignore internal compiler-generated variables
                if not var_name.startswith("_"):
                    is_failed = self._contains_sorry(node.get("val", {}))
                    # A letE binder's variable is accessed as bvar idx 0 inside its immediate body
                    is_used = self._is_bvar_used(node.get("body", {}), target_idx=0)

                    if is_used and not is_failed:
                        results["core_solved"].append(var_name)
                    elif is_used and is_failed:
                        results["core_failed"].append(var_name)
                    elif not is_used and is_failed:
                        results["malignant"].append(var_name)
                    elif not is_used and not is_failed:
                        results["benign"].append(var_name)

            # Continue traversing down the tree to find nested letE nodes
            for val in node.values():
                if isinstance(val, dict):
                    traverse(val)

        traverse(root_expr)
        return results

    def get_unused_context_variables(self, root_expr: Dict[str, Any]) -> List[str]:
        """
        Scans the root theorem parameters (`forallE` or `lam`) to find hypotheses 
        that are never used in the proof body. Useful for Context Pruning before LLM.
        """
        unused_vars = []
        current_node = root_expr

        # Unroll the top-level parameters of the theorem
        while isinstance(current_node, dict) and current_node.get("expr") in ("forallE", "lam"):
            var_name = current_node.get("var_name", "unknown")
            body = current_node.get("body", {})
            
            # If the parameter is not used anywhere in the rest of the theorem
            if not self._is_bvar_used(body, target_idx=0):
                if not var_name.startswith("_"):
                    unused_vars.append(var_name)
            
            current_node = body

        return unused_vars