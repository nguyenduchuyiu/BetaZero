from __future__ import annotations
from typing import Dict, List, Any

class ExprDependencyAnalyzer:
    def _contains_sorry(self, node: Any) -> bool:
        if not isinstance(node, dict): return False
        if node.get("expr") == "const" and node.get("name") in ("sorryAx", "sorry"): return True
        return any(self._contains_sorry(v) for k, v in node.items() if k != "expr")

    def _is_bvar_used(self, node: Any, target_idx: int) -> bool:
        if not isinstance(node, dict): return False
        expr_type = node.get("expr")
        if expr_type == "bvar":
            return node.get("idx") == target_idx
        if expr_type in ("lam", "forallE", "letE"):
            # Binders: check type/val (idx giữ nguyên), check body (idx + 1)
            return (self._is_bvar_used(node.get("var_type"), target_idx) or 
                    self._is_bvar_used(node.get("val"), target_idx) or 
                    self._is_bvar_used(node.get("body"), target_idx + 1))
        return any(self._is_bvar_used(v, target_idx) for k, v in node.items() if k != "expr")

    def classify_skeleton_subgoals(self, root_expr: Dict[str, Any], allowed_vars: set[str] | None = None) -> Dict[str, List[str]]:
        results = {"core_solved": [], "core_failed": [], "malignant": [], "benign": []}

        def traverse(node: Any):
            if not isinstance(node, dict): return
            
            # CASE 1: Node là letE (Dạng have tường minh)
            if node.get("expr") == "letE":
                self._classify(node, node.get("val"), results, allowed_vars)
            
            # CASE 2: Node là App(Lam, Val) - Dạng have bị convert
            elif node.get("expr") == "app" and isinstance(node.get("fn"), dict) and node["fn"].get("expr") == "lam":
                lam_node = node["fn"]
                val_node = node.get("arg") # Giá trị truyền vào chính là proof của subgoal
                self._classify(lam_node, val_node, results, allowed_vars)

            for k, v in node.items():
                if k != "expr" and isinstance(v, dict): traverse(v)

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
            # Nếu có danh sách biến hợp lệ, bỏ qua các biến cục bộ không nằm trong danh sách
            if allowed_vars is not None and var_name not in allowed_vars:
                return
                
            # Biến là bvar 0 bên trong body của chính nó
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
                
                current_node = body # Tiếp tục đi sâu vào chuỗi tham số
            
            elif expr_type == "mdata":
                current_node = current_node.get("inner", {})
                
            else:
                # Đã thoát khỏi chuỗi tham số đầu vào (ví dụ gặp body chính, letE, app, ...)
                # Dừng lại, không quét tiếp vào sâu bên trong nữa để tránh bắt nhầm biến nội bộ.
                break

        return unused_vars
    
SHARED_EXPR_ANALYZER = ExprDependencyAnalyzer()