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
from analysis import figures, loaders, metrics


ROLLOUT = Path("outputs/rollouts/gemini3flash/miniF2F-test/amc12b_2002_p11.json")


def test_parse_real_rollout_and_detect_failed_full_reward():
    data = json.loads(ROLLOUT.read_text(encoding="utf-8"))
    analysis = analyze_rollout_data(data, theorem=ROLLOUT.name, source_path=str(ROLLOUT))

    theorem = analysis["theorem_rows"][0]
    assert theorem["total_nodes"] == 551
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


def test_analysis_loader_counts_only_root_status_as_solved(tmp_path):
    rollout = {
        "root_id": "state_0",
        "search_metadata": {
            "budget": {"max_nodes": 8, "used_total": 8, "lean_verify_calls": 8, "patch_verify_calls": 1},
            "final_status": {"states": {"SOLVED": 1, "FAILED": 1}, "actions": {}},
            "depth_distribution": {"states_solved_by_depth": {"1": 1}, "max_depth_reached": 1},
        },
        "nodes": [
            {"id": "state_0", "type": "OR", "status": "FAILED", "depth": 0, "content": {"proof_body": "sorry"}},
            {"id": "action_0", "type": "AND", "action_type": "skeleton", "status": "FAILED", "metrics": {"r_env": 1.0, "r_dep": 0.0, "Q_value": 1.0}},
            {"id": "state_1", "type": "OR", "status": "SOLVED", "depth": 1, "content": {"proof_body": "rfl"}},
        ],
        "edges": [
            {"source": "state_0", "target": "action_0", "relation": "expanded_to"},
            {"source": "action_0", "target": "state_1", "relation": "subgoal"},
        ],
    }
    path = tmp_path / "child_solved_root_failed.json"
    path.write_text(json.dumps(rollout), encoding="utf-8")

    rec = loaders.load_rollout_file(str(path))

    assert rec["root_status"] == "FAILED"
    assert rec["solved"] is False
    assert metrics.solve_rate([rec])["solved"] == 0


def test_analysis_hierarchy_separates_root_only_and_skeleton(tmp_path):
    root_only = {
        "root_id": "state_0",
        "search_metadata": {
            "budget": {},
            "final_status": {"states": {"SOLVED": 1}, "actions": {}},
            "depth_distribution": {"states_solved_by_depth": {"0": 1}, "max_depth_reached": 0},
        },
        "nodes": [{"id": "state_0", "type": "OR", "status": "SOLVED", "depth": 0, "content": {"proof_body": "rfl"}}],
        "edges": [],
    }
    via_skeleton = {
        "root_id": "state_0",
        "search_metadata": {
            "budget": {},
            "final_status": {"states": {"SOLVED": 2}, "actions": {}},
            "depth_distribution": {"states_solved_by_depth": {"1": 1}, "max_depth_reached": 1},
        },
        "nodes": [
            {"id": "state_0", "type": "OR", "status": "SOLVED", "depth": 0, "content": {"proof_body": "exact h"}},
            {"id": "state_1", "type": "OR", "status": "SOLVED", "depth": 1, "content": {"proof_body": "rfl"}},
        ],
        "edges": [],
    }
    p1 = tmp_path / "root_only.json"
    p2 = tmp_path / "via_skeleton.json"
    p1.write_text(json.dumps(root_only), encoding="utf-8")
    p2.write_text(json.dumps(via_skeleton), encoding="utf-8")

    hb = metrics.hierarchical_breakdown([
        loaders.load_rollout_file(str(p1)),
        loaders.load_rollout_file(str(p2)),
    ])

    assert hb["solved"] == 2
    assert hb["solved_root_only"] == 1
    assert hb["solved_via_skeleton"] == 1


def test_analysis_tables_tolerate_empty_reward_inputs(tmp_path):
    sep = metrics.reward_separability([])
    root_q = metrics.root_q_vs_outcome([])

    figures.write_reward_table(str(tmp_path), sep, root_q)
    figures.write_trajectory_table(
        str(tmp_path),
        metrics.trajectory_stats([]),
        metrics.graph_structure([]),
        metrics.reward_disagreement([]),
    )

    assert (tmp_path / "table_reward.md").exists()
    assert (tmp_path / "table_trajectory.md").exists()
