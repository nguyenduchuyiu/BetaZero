import json
import os
from typing import Any, Dict

from betazero.core import ProofState, Action
from betazero.search.graph import ANDORGraph

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

        def traverse_state(state: ProofState):
            if state in visited_states:
                return
            visited_states.add(state)
            
            s_id = self._get_state_id(state)
            nodes.append({
                "id": s_id,
                "type": "OR",
                "status": graph.status(state),
                "depth": graph.get_depth(state),
                "content": {
                    "context": state.context,
                    "goal": state.goal
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

            nodes.append({
                "id": a_id,
                "type": "AND",
                "action_type": action.action_type,
                "is_sc_tactic": action.is_sc_tactic,
                "status": graph.status(action),
                "content": action.content,
                "prompt": action.prompt,
                "extracted_lean_code": extracted_lean_code,
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