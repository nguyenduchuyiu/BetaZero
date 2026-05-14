import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_rollouts import (
    analyze_rollout_data,
    detect_cycles,
    render_dot,
    write_summary,
    write_dot_graph,
)


ROLLOUT = Path("outputs/rollouts/gemini3flash/miniF2F-valid-50/aime_1983_p2.json")


def test_parse_real_rollout_and_detect_failed_full_reward():
    data = json.loads(ROLLOUT.read_text(encoding="utf-8"))
    analysis = analyze_rollout_data(data, theorem=ROLLOUT.name, source_path=str(ROLLOUT))

    theorem = analysis["theorem_rows"][0]
    assert theorem["total_nodes"] == 236
    assert theorem["or_nodes"] > 0
    assert theorem["and_nodes"] > 0
    assert theorem["tactic_actions"] > 0
    assert theorem["skeleton_actions"] > 0

    kinds = {row["kind"] for row in analysis["anomaly_rows"]}
    assert "failed_full_r_env" in kinds


def test_cycle_detection_reports_repeated_path():
    data = {
        "root_id": "state_0",
        "nodes": [
            {"id": "state_0", "type": "OR", "status": "OPEN", "depth": 0, "content": {"goal": "A", "context": "", "proof_body": "  sorry"}, "metrics": {"V_value": 0}},
            {"id": "action_0", "type": "AND", "action_type": "skeleton", "status": "OPEN", "content": "", "prompt": "", "extracted_lean_code": "have h : A := sorry", "metrics": {"r_env": 1.0, "r_dep": 0.0, "Q_value": 1.0}},
        ],
        "edges": [
            {"source": "state_0", "target": "action_0", "relation": "expanded_to"},
            {"source": "action_0", "target": "state_0", "relation": "subgoal"},
        ],
    }
    analysis = analyze_rollout_data(data, theorem="cycle.json")
    assert analysis["theorem_rows"][0]["cycles"] == 1
    assert any(row["kind"] == "cycle_detected" for row in analysis["anomaly_rows"])

    nodes = {n["id"]: n for n in data["nodes"]}
    cycles = detect_cycles(nodes, data["edges"])
    assert len(cycles) == 1
    assert "state_0" in cycles[0]["cycle_nodes"]


def test_acyclic_rollout_has_no_cycles():
    data = json.loads(ROLLOUT.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    assert detect_cycles(nodes, data["edges"]) == []


def test_dot_export_contains_publication_layout_and_renders_when_dot_exists(tmp_path):
    data = {
        "root_id": "state_0",
        "nodes": [
            {"id": "state_0", "type": "OR", "status": "SOLVED", "depth": 0, "content": {"goal": "A", "context": "", "proof_body": "rfl"}, "metrics": {"V_value": 2.0}},
            {"id": "action_0", "type": "AND", "action_type": "tactic", "status": "SOLVED", "content": "", "prompt": "", "extracted_lean_code": "rfl", "metrics": {"r_env": 1.0, "r_dep": 0.0, "Q_value": 2.0}},
        ],
        "edges": [
            {"source": "state_0", "target": "action_0", "relation": "expanded_to"},
        ],
    }
    analysis = analyze_rollout_data(data, theorem="tiny.json")
    dot_path = tmp_path / "tiny.dot"
    write_dot_graph(dot_path, analysis)
    dot_text = dot_path.read_text(encoding="utf-8")

    assert "rankdir=TB" in dot_text
    assert "shape=ellipse" in dot_text
    assert "shape=box" in dot_text

    if shutil.which("dot"):
        png_path = render_dot(dot_path)
        assert png_path is not None
        assert png_path.exists()


def test_summary_explains_failure_modes_instead_of_examples_and_recommendations(tmp_path):
    data = json.loads(ROLLOUT.read_text(encoding="utf-8"))
    analysis = analyze_rollout_data(data, theorem=ROLLOUT.name, source_path=str(ROLLOUT))
    summary_path = tmp_path / "summary.md"

    write_summary(
        summary_path,
        pd.DataFrame(analysis["theorem_rows"]),
        pd.DataFrame(analysis["node_rows"]),
        pd.DataFrame(analysis["anomaly_rows"]),
    )

    text = summary_path.read_text(encoding="utf-8")
    assert "Top examples:" not in text
    assert "## Recommendations" not in text
    assert "`failed_full_r_env`" in text
    assert "environment/repair reward" in text
