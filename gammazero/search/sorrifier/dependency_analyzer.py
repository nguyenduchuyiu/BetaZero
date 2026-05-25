from __future__ import annotations
from typing import Dict, List, Any

class ExprDependencyAnalyzer:
    def _contains_sorry(self, node: Any) -> bool:
        stack = [node]
        while stack:
            curr = stack.pop()
            if not isinstance(curr, dict): continue
            if curr.get("expr") == "const" and curr.get("name") in ("sorryAx", "sorry"):
                return True
            for k, v in curr.items():
                if k != "expr": stack.append(v)
        return False

    def _is_bvar_used(self, node: Any, target_idx: int) -> bool:
        stack = [(node, target_idx)]
        while stack:
            curr, idx = stack.pop()
            if not isinstance(curr, dict): continue
            expr_type = curr.get("expr")
            if expr_type == "bvar":
                if curr.get("idx") == idx: return True
                continue
            if expr_type in ("lam", "forallE", "letE"):
                stack.append((curr.get("var_type"), idx))
                stack.append((curr.get("val"), idx))
                stack.append((curr.get("body"), idx + 1))
            else:
                for k, v in curr.items():
                    if k != "expr": stack.append((v, idx))
        return False

    def _find_decl_value(self, root_node: Any, target_name: str) -> Any | None:
        stack = [root_node]
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue

            if node.get("expr") == "letE" and node.get("var_name") == target_name:
                return node.get("val")

            if (
                node.get("expr") == "app"
                and isinstance(node.get("fn"), dict)
                and node["fn"].get("expr") == "lam"
                and node["fn"].get("var_name") == target_name
            ):
                return node.get("arg")

            for k, v in node.items():
                if k != "expr" and isinstance(v, dict):
                    stack.append(v)
        return None

    def classify_skeleton_subgoals(
        self,
        root_expr: Dict[str, Any],
        allowed_vars: set[str] | None = None,
        target_name: str | None = None,
    ) -> Dict[str, List[str]]:
        results = {"core_solved": [], "core_failed": [], "malignant": [], "benign": []}

        def traverse(root_node: Any):
            stack = [root_node]
            while stack:
                node = stack.pop()
                if not isinstance(node, dict): continue
                
                # CASE 1: explicit `have` (letE node).
                if node.get("expr") == "letE":
                    self._classify(node, node.get("val"), results, allowed_vars)
                
                # CASE 2: App(Lam, Val), the `have` form Lean 4 lowers it into.
                elif node.get("expr") == "app" and isinstance(node.get("fn"), dict) and node["fn"].get("expr") == "lam":
                    lam_node = node["fn"]
                    val_node = node.get("arg")  # the proof of the subgoal
                    self._classify(lam_node, val_node, results, allowed_vars)

                for k, v in node.items():
                    if k != "expr" and isinstance(v, dict): stack.append(v)

        if target_name:
            target_val = self._find_decl_value(root_expr, target_name)
            if target_val is None:
                return results
            if self._contains_sorry(target_val):
                results["core_failed"].append("MAIN_GOAL")
            else:
                results["core_solved"].append("MAIN_GOAL")
            traverse(target_val)
            return {k: list(set(v)) for k, v in results.items()}

        # Check if the main goal itself is closed by a naked 'sorry'
        # We traverse the root to find the innermost body
        current = root_expr
        while isinstance(current, dict):
            if current.get("expr") in ("lam", "forallE", "letE"):
                current = current.get("body")
            elif current.get("expr") == "mdata":
                current = current.get("inner")
            elif current.get("expr") == "app":
                # For app, if it's a let-binding equivalent, we follow the body.
                # If not, the main goal is just an application.
                fn = current.get("fn")
                if isinstance(fn, dict) and fn.get("expr") == "lam":
                    current = fn.get("body")
                else:
                    break
            else:
                break
        
        # If the final unpacked body contains sorry, the main goal is failed!
        if self._contains_sorry(current):
            results["core_failed"].append("MAIN_GOAL")
        else:
            results["core_solved"].append("MAIN_GOAL")

        traverse(root_expr)
        return {k: list(set(v)) for k, v in results.items()}

    def _classify(self, binder_node: dict, val_node: Any, results: dict, allowed_vars: set[str] | None = None):
        var_name = binder_node.get("var_name", "")
        if var_name and not var_name.startswith("_"):
            # When `allowed_vars` is provided, skip locals outside that set.
            if allowed_vars is not None and var_name not in allowed_vars:
                return
                
            # The bound variable appears as bvar 0 inside its own body.
            is_used = self._is_bvar_used(binder_node.get("body"), 0)
            is_failed = self._contains_sorry(val_node)
            
            if is_used:
                results["core_failed" if is_failed else "core_solved"].append(var_name)
            else:
                results["malignant" if is_failed else "benign"].append(var_name)

    def get_unused_context_variables(self, root_expr: Dict[str, Any]) -> List[str]:
        unused_vars = []
        current_node = root_expr

        # Unroll the top-level parameters of the theorem sequentially
        while isinstance(current_node, dict):
            expr_type = current_node.get("expr")
            
            if expr_type in ("forallE", "lam"):
                var_name = current_node.get("var_name", "")
                body = current_node.get("body", {})
                
                # We usually ignore internal compiler-generated variables
                if var_name and not var_name.startswith("_"):
                    # If the parameter is not used anywhere in the rest of the theorem
                    if not self._is_bvar_used(body, target_idx=0):
                        unused_vars.append(var_name)
                
                current_node = body  # walk further into the parameter chain
            
            elif expr_type == "mdata":
                current_node = current_node.get("inner", {})
                
            else:
                # We have left the input parameter chain (body, letE, app, ...).
                # Stop here; descending further would catch internal binders.
                break

        return unused_vars
    
SHARED_EXPR_ANALYZER = ExprDependencyAnalyzer()
