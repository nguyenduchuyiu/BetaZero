from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import uuid


class NodeStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DECOMPOSED = "decomposed"
    SOLVED = "solved"
    FAILED = "failed"
    ESCALATED = "escalated"


class RouteDecision(str, Enum):
    DECOMPOSE = "DECOMPOSE"
    SOLVE = "SOLVE"


@dataclass
class SubgoalNode:
    """Một nút trong cây subgoal (Layer 2/3)."""

    id: str
    formal_statement: str
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    resolved_proof: Optional[str] = None
    router_decision: Optional[RouteDecision] = None
    retry_count: int = 0
    last_error_summary: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerOutput:
    informal_plan: str
    skeleton_code: str


@dataclass
class ProofState:
    """Context store cho orchestrator: cây subgoal + lịch sử."""

    root_statement: str
    nodes: dict[str, SubgoalNode] = field(default_factory=dict)
    root_ids: list[str] = field(default_factory=list)
    planner_output: Optional[PlannerOutput] = None
    initial_repl_snapshot: Optional[dict[str, Any]] = None
    skeleton_repl_snapshot: Optional[dict[str, Any]] = None
    final_proof: Optional[str] = None
    event_log: list[dict[str, Any]] = field(default_factory=list)

    def new_node(self, formal_statement: str, parent_id: Optional[str] = None) -> SubgoalNode:
        nid = str(uuid.uuid4())
        node = SubgoalNode(id=nid, formal_statement=formal_statement, parent_id=parent_id)
        self.nodes[nid] = node
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(nid)
        else:
            self.root_ids.append(nid)
        return node

    def log(self, event: str, **data: Any) -> None:
        self.event_log.append({"event": event, **data})
