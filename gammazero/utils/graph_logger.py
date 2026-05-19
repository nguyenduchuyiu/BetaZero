import json
import os
from typing import Any, Dict

from gammazero.core import ProofState, Action
from gammazero.search.graph import ANDORGraph
from gammazero.utils.scaffold import target_subgoal_label

class GraphLogger:
    """Crawler đồ thị AND/OR để export ra JSON dùng cho visualization."""

    def __init__(self):
        self.obj_to_id: Dict[Any, str] = {}
        self.state_counter = 0
        self.action_counter = 0

    def _get_state_id(self, state: ProofState) -> str:
        if state not in self.obj_to_id:
            self.obj_to_id[state] = f"state_{self.state_counter}"
            self.state_counter += 1
        return self.obj_to_id[state]

    def _get_action_id(self, action: Action) -> str:
        if action not in self.obj_to_id:
            self.obj_to_id[action] = f"action_{self.action_counter}"
            self.action_counter += 1
        return self.obj_to_id[action]

    def export_to_dict(self, graph: ANDORGraph, root: ProofState, q_values: Dict[Action, float]) -> Dict[str, Any]:
        """Duyệt đồ thị và build cấu trúc Flat (nodes, edges)."""
        nodes = []
        edges = []
        
        visited_states = set()
        visited_actions = set()

        def is_generic_label(state: ProofState, label: str | None) -> bool:
            if not label or label == f"target_{state.target_index}":
                return True
            kind = (state.target_kind or "").strip()
            return bool(kind and label == f"{kind}_{state.target_index}")

        def label_from_parent_skeleton(
            state: ProofState,
            child_index_filter: int | None = None,
        ) -> str | None:
            for _, skeleton_action, child_index in graph.parent_skeleton_items_for_state(state):
                if child_index_filter is not None and child_index != child_index_filter:
                    continue
                return target_subgoal_label(
                    skeleton_action.extracted_code,
                    child_index,
                    target_kind="skeleton_child",
                )
            return None

        def state_target_label(state: ProofState) -> str:
            label = target_subgoal_label(
                state.scaffold_code,
                state.target_index,
                target_kind=state.target_kind,
            )
            if is_generic_label(state, label):
                return label_from_parent_skeleton(state) or label
            return label

        def traverse_state(state: ProofState):
            if state in visited_states:
                return
            visited_states.add(state)
            
            s_id = self._get_state_id(state)
            target_label = state_target_label(state)
            # Trích xuất proof body nếu trạng thái đã SOLVED
            proof_body = (
                graph.extract_proof_code(state)
                if graph.status(state) == "SOLVED"
                else None
            ) or "  sorry"

            nodes.append({
                "id": s_id,
                "type": "OR",
                "status": graph.status(state),
                "depth": graph.get_depth(state),
                "target_label": target_label,
                "content": {
                    "context": state.context,
                    "goal": state.goal,
                    "proof_body": proof_body,
                    "scaffold_code": state.scaffold_code,
                    "target_index": state.target_index,
                    "target_kind": state.target_kind,
                    "target_label": target_label,
                    "parent_action_id": state.parent_action_id,
                },
                "metrics": {
                    "V_value": max(
                        (
                            graph.backup_value_for_action(a, q_values.get(a, 0.0))
                            for a in graph.get_actions(state)
                        ),
                        default=0.0,
                    )
                }
            })

            for action in graph.get_actions(state):
                a_id = self._get_action_id(action)
                edges.append({"source": s_id, "target": a_id, "relation": "expanded_to"})
                traverse_action(action)

        def traverse_action(action: Action):
            if action in visited_actions:
                return
            visited_actions.add(action)

            a_id = self._get_action_id(action)
            # Truy cập internal _r_dep (nếu ông chưa viết hàm get_r_dep trong ANDORGraph)
            r_d = graph._r_dep.get(action, 0.0) 
            prompt = action.prompt or ""
            extracted_lean_code = action.extracted_code
            parent_state = graph.get_parent(action)
            target_label = None
            if parent_state is not None:
                target_label = state_target_label(parent_state)
            target_child_label = None
            if action.target_child_index is not None and parent_state is not None:
                target_child_label = label_from_parent_skeleton(
                    parent_state,
                    child_index_filter=action.target_child_index,
                )
                if target_child_label is None:
                    target_child_label = target_label

            nodes.append({
                "id": a_id,
                "internal_id": action.id,  # THÊM ID GỐC ĐỂ TRACE
                "type": "AND",
                "action_type": action.action_type,
                "status": graph.status(action),
                "content": action.content,
                "prompt": action.prompt,
                "extracted_lean_code": extracted_lean_code,
                "verify_code": action.verify_code,
                "stitched_code": action.stitched_code,
                "patched_code": action.patched_code,
                "lean_feedback": action.lean_feedback,
                "target_label": target_label,
                "target_child_index": action.target_child_index,
                "target_child_label": target_child_label,
                "metrics": {
                    "r_env": graph.get_r_env(action),
                    "r_dep": r_d,
                    "Q_value": q_values.get(action, 0.0)
                }
            })

            for child_state in action.children:
                c_id = self._get_state_id(child_state)
                edges.append({"source": a_id, "target": c_id, "relation": "subgoal"})
                traverse_state(child_state)

        # Bắt đầu duyệt từ root
        traverse_state(root)

        return {
            "theorem_goal": root.goal,
            "root_id": self._get_state_id(root),
            "total_nodes": len(nodes),
            "search_metadata": getattr(graph, "search_metadata", None),
            "nodes": nodes,
            "edges": edges
        }

    def save_json(self, graph: ANDORGraph, root: ProofState, q_values: Dict[Action, float], filepath: str):
        """Export và lưu thành file JSON."""
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        data = self.export_to_dict(graph, root, q_values)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
