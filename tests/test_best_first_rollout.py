from __future__ import annotations

from gammazero.core import Action, ProofState
from gammazero.policy.prompt import build_skeleton_retry_prompt
from gammazero.search.graph import ANDORGraph
from gammazero.search.rollout.heuristic import SimpleHeuristicScorer
from gammazero.search.rollout.best_first_rollout import BestFirstRollout
from gammazero.search.rollout.search_queue import StatePriorityQueue
from gammazero.search.rollout.search_stats import StateStats


class FakePolicy:
    def __init__(self):
        self.calls: list[tuple[str, tuple[str, ...], int]] = []

    def sample(self, states, action_type, n, *, prompts=None):
        self.calls.append((action_type, tuple(state.goal for state in states), n))
        return [
            [{"text": f"{action_type}:{state.goal}:{i}"} for i in range(n)]
            for state in states
        ]


class FakeReward:
    def compute_returns(self, graph):
        return graph.backup()

    def r_env(self, *args, **kwargs):
        return 1.0


class NoopAssigner:
    def assign(self, graph):
        pass

    def stitch_and_score_skeletons(self, graph):
        pass


class ScriptedExecutor:
    def __init__(self, tactic_status, skeleton_children):
        self.tactic_status = tactic_status
        self.skeleton_children = skeleton_children

    def execute(self, graph, states, batches, action_type, budget, prompts=None):
        for state, batch in zip(states, batches):
            for item in batch:
                if not budget.try_consume():
                    return []
                if action_type == "tactic":
                    status = self.tactic_status(state)
                    graph.expand(
                        state,
                        Action("tactic", item["text"], extracted_code="tactic"),
                        r_env=1.0 if status == "SOLVED" else 0.0,
                        tactic_status=status,
                    )
                else:
                    children = self.skeleton_children(state)
                    action = Action(
                        "skeleton",
                        item["text"],
                        extracted_code="have h := by sorry",
                        children=children,
                    )
                    graph.expand(state, action, r_env=1.0)
        return []


class NoopExecutor:
    def execute(self, graph, states, batches, action_type, budget, prompts=None):
        return []


class FeedbackExecutor:
    def execute(self, graph, states, batches, action_type, budget, prompts=None):
        return [[("have h : False := by exact bad", "unknown identifier 'bad'", "")]]


class FixedScoreScorer:
    def __init__(self, skeleton_scores):
        self.skeleton_scores = skeleton_scores

    def score_state(self, state, graph, stats):
        return 0.0

    def score_skeleton(self, action, parent_state, graph, stats):
        return self.skeleton_scores[action.content]


def make_rollout(policy=None, executor=None, **kwargs):
    defaults = dict(
        max_depth=1,
        max_nodes=32,
        initial_tactic_k=1,
        retry_tactic_k=1,
        max_tactic_per_state=1,
        initial_skeleton_k=1,
        retry_skeleton_k=1,
        max_skeleton_per_state=1,
        search_batch_size=4,
        state_beam_width=32,
        state_beam_per_depth=8,
        skeleton_beam_per_state=1,
    )
    defaults.update(kwargs)
    return BestFirstRollout(
        policy or FakePolicy(),
        None,
        None,
        FakeReward(),
        executor=executor or NoopExecutor(),
        reward_assigner=NoopAssigner(),
        **defaults,
    )


def test_case_1_root_tactic_solve_stops_before_skeleton():
    policy = FakePolicy()
    root = ProofState("", "root")
    executor = ScriptedExecutor(
        tactic_status=lambda state: "SOLVED",
        skeleton_children=lambda state: (),
    )
    rollout = make_rollout(policy=policy, executor=executor)

    _, graph, _ = rollout.rollout(root)

    actions = graph.get_actions(root)
    assert graph.status(root) == "SOLVED"
    assert len(actions) == 1
    assert actions[0].action_type == "tactic"
    assert graph.status(actions[0]) == "SOLVED"
    assert [call[0] for call in policy.calls] == ["tactic"]


def test_solved_tactic_backup_uses_r_dep_instead_of_w_solve():
    root = ProofState("", "root")
    graph = ANDORGraph(root)
    tactic = Action("tactic", "solve", extracted_code="trivial")
    graph.expand(root, tactic, r_env=1.0, r_dep=0.25, tactic_status="SOLVED")

    q_values = graph.backup(W_solve=99.0)

    assert q_values[tactic] == 1.25


def test_case_2_skeleton_children_both_tactic_solve_root_solved():
    root = ProofState("", "root")
    child1 = ProofState("", "child1")
    child2 = ProofState("", "child2")
    executor = ScriptedExecutor(
        tactic_status=lambda state: "SOLVED" if state.goal.startswith("child") else "FAILED",
        skeleton_children=lambda state: (child1, child2) if state == root else (),
    )
    rollout = make_rollout(executor=executor)

    _, graph, _ = rollout.rollout(root)

    root_actions = graph.get_actions(root)
    skeletons = [action for action in root_actions if action.action_type == "skeleton"]
    assert graph.status(root) == "SOLVED"
    assert len(skeletons) == 1
    assert graph.status(skeletons[0]) == "SOLVED"
    assert graph.status(child1) == "SOLVED"
    assert graph.status(child2) == "SOLVED"


def test_case_3_failed_child_fails_skeleton_but_parent_can_remain_open():
    root = ProofState("", "root")
    child = ProofState("", "child")
    graph = ANDORGraph(root)
    skeleton = Action("skeleton", "split", children=(child,))
    graph.expand(root, skeleton, r_env=1.0)
    graph.add_state(child, depth=1)
    graph.mark_failed(child)

    rollout = make_rollout()
    stats = {
        root: StateStats(depth=0, tactic_tries=0, skeleton_tries=1, tactic_probe_done=True),
        child: StateStats(depth=1, exhausted=True),
    }
    rollout.propagate(graph, stats)

    assert graph.status(child) == "FAILED"
    assert graph.status(skeleton) == "FAILED"
    assert graph.status(root) == "OPEN"


def test_case_4_queue_prune_and_duplicate_child_activation():
    root = ProofState("", "root")
    duplicate = ProofState("", "same")
    action = Action("skeleton", "split", children=(duplicate, duplicate))
    graph = ANDORGraph(root)
    stats = {root: StateStats(depth=0)}
    queue = StatePriorityQueue()
    rollout = make_rollout(state_beam_width=3, state_beam_per_depth=2)

    rollout.activate_skeleton_children(
        [(root, action, 0.0)],
        graph,
        stats,
        queue,
        seen_states={rollout.state_key(root)},
    )

    assert len([state for _, state in queue.items() if state == duplicate]) == 1

    for i in range(8):
        state = ProofState("", f"s{i}")
        graph.add_state(state, depth=i % 2)
        stats[state] = StateStats(depth=i % 2)
        queue.push(state, float(i))

    rollout.prune_queue(queue, graph, stats)
    kept = queue.items()
    per_depth = {}
    for _, state in kept:
        per_depth[stats[state].depth] = per_depth.get(stats[state].depth, 0) + 1

    assert len(kept) <= 3
    assert all(count <= 2 for count in per_depth.values())
    assert len({rollout.state_key(state) for _, state in kept}) == len(kept)


def test_case_5_max_depth_allows_tactic_but_blocks_skeleton():
    root = ProofState("", "root")
    graph = ANDORGraph(root)
    rollout = make_rollout(max_depth=1)
    stats = {
        root: StateStats(
            depth=1,
            tactic_tries=0,
            skeleton_tries=0,
            tactic_probe_done=True,
        )
    }

    assert rollout.should_try_tactic(root, graph, stats)
    assert not rollout.should_expand_skeleton(root, graph, stats)


def test_duplicate_only_skeleton_round_exhausts_skeleton_lane_not_tactic_lane():
    root = ProofState("", "root")
    duplicate = ProofState("", "same")
    action = Action("skeleton", "split", children=(duplicate, duplicate))
    graph = ANDORGraph(root)
    stats = {
        root: StateStats(
            depth=0,
            tactic_tries=1,
            skeleton_tries=1,
            tactic_probe_done=True,
            skeleton_probe_done=True,
        )
    }
    queue = StatePriorityQueue()
    rollout = make_rollout(max_tactic_per_state=2, max_skeleton_per_state=2)
    seen = {rollout.state_key(root), rollout.state_key(duplicate)}

    child_stats = rollout.activate_skeleton_children([(root, action, 0.0)], graph, stats, queue, seen)
    rollout.update_skeleton_progress([(root, action, 0.0)], child_stats["generated_by_parent"], stats)

    assert child_stats["generated"] == 0
    assert child_stats["duplicates"] == 2
    assert stats[root].last_skeleton_new_children == 0
    assert stats[root].bad_skeleton_rounds == 1
    assert stats[root].skeleton_exhausted
    assert not rollout.should_expand_skeleton(root, graph, stats)
    assert rollout.should_try_tactic(root, graph, stats)


def test_budget_partial_execution_counts_only_inserted_actions():
    root = ProofState("", "root")
    other = ProofState("", "other")
    graph = ANDORGraph(root)
    graph.add_state(other, depth=0)
    stats = {root: StateStats(depth=0), other: StateStats(depth=0)}
    executor = ScriptedExecutor(
        tactic_status=lambda state: "FAILED",
        skeleton_children=lambda state: (),
    )
    rollout = make_rollout(
        executor=executor,
        max_nodes=2,
        initial_tactic_k=4,
        max_tactic_per_state=4,
    )

    jobs = rollout.make_tactic_jobs([root, other], graph, stats)
    counts, action_stats = rollout.run_jobs(graph, jobs, "tactic")
    for state, count in counts.items():
        stats[state].tactic_tries += count

    assert jobs == [(root, 2), (other, 2)]
    assert len(graph.get_actions(root)) + len(graph.get_actions(other)) == 2
    assert stats[root].tactic_tries + stats[other].tactic_tries == 2
    assert max(stats[root].tactic_tries, stats[other].tactic_tries) <= 2
    assert action_stats["budget_used"] == 2


def test_tactic_r_env_updates_state_score_signal():
    root = ProofState("", "root")
    graph = ANDORGraph(root)
    weak = Action("tactic", "weak", extracted_code="weak")
    strong = Action("tactic", "strong", extracted_code="strong")
    graph.expand(root, weak, r_env=0.25, tactic_status="FAILED")
    graph.expand(root, strong, r_env=0.75, tactic_status="FAILED")
    stats = {root: StateStats(depth=0)}
    rollout = make_rollout()

    rollout.update_best_tactic_r_env(graph, [weak, strong], stats)
    score = rollout.scorer.score_state(root, graph, stats)

    assert stats[root].best_tactic_r_env == 0.75
    assert score == 1.5 * 0.75


def test_invalid_skeleton_failed_and_parent_open_if_retry_budget_remains():
    root = ProofState("", "root")
    graph = ANDORGraph(root)
    invalid = Action("skeleton", "bad skeleton", children=())
    graph.expand(root, invalid, r_env=0.0)
    stats = {
        root: StateStats(
            depth=0,
            tactic_tries=1,
            skeleton_tries=1,
            tactic_probe_done=True,
            skeleton_probe_done=True,
        )
    }
    rollout = make_rollout(max_tactic_per_state=2, max_skeleton_per_state=2)

    valid = rollout.valid_skeletons_from_actions(graph, [invalid])
    rollout.propagate(graph, stats)

    assert valid == []
    assert graph.status(invalid) == "FAILED"
    assert graph.status(root) == "OPEN"
    assert rollout.can_requeue_state(root, graph, stats)


def test_valid_zero_children_excludes_raw_failed_skeletons():
    root = ProofState("", "root")
    graph = ANDORGraph(root)
    raw_failed = Action("skeleton", "raw failed", children=())
    raw_valid_zero = Action("skeleton", "raw valid zero", children=())
    graph.expand(root, raw_failed, r_env=0.0)
    graph.mark_failed(raw_failed)
    graph.expand(root, raw_valid_zero, r_env=1.0)
    graph.mark_solved(raw_valid_zero)
    rollout = make_rollout()

    zero_children = sum(
        1 for action in [raw_failed, raw_valid_zero]
        if action.action_type == "skeleton"
        and not rollout.is_sorrified_action(action)
        and graph.status(action) != "FAILED"
        and len(action.children) == 0
    )

    assert zero_children == 1


def test_skeleton_feedback_is_cached_and_added_to_retry_prompt():
    root = ProofState("", "root")
    graph = ANDORGraph(root)
    rollout = make_rollout(executor=FeedbackExecutor())

    rollout.run_jobs(graph, [(root, 1)], "skeleton")
    prompt = build_skeleton_retry_prompt(root, rollout._skeleton_feedback_by_state[root])

    assert "PREVIOUS SKELETON ATTEMPTS FAILED" in prompt
    assert "FAILED SKELETON CODE" in prompt
    assert "unknown identifier 'bad'" in prompt
    assert "Do not repeat the failed skeleton" in prompt


def test_parent_requeues_after_failed_attempts_when_budget_remains():
    root = ProofState("", "root")
    graph = ANDORGraph(root)
    failed_tactic = Action("tactic", "bad tactic", extracted_code="bad")
    failed_skeleton = Action("skeleton", "bad skeleton", children=())
    graph.expand(root, failed_tactic, tactic_status="FAILED")
    graph.expand(root, failed_skeleton)
    graph.mark_failed(failed_skeleton)
    stats = {
        root: StateStats(
            depth=0,
            tactic_tries=1,
            skeleton_tries=1,
            tactic_probe_done=True,
            skeleton_probe_done=True,
        )
    }
    queue = StatePriorityQueue()
    rollout = make_rollout(max_tactic_per_state=2, max_skeleton_per_state=2)

    if rollout.can_requeue_state(root, graph, stats):
        score = rollout.scorer.score_state(root, graph, stats)
        stats[root].last_score = score
        queue.push(root, score)

    assert graph.status(root) == "OPEN"
    assert len(queue) == 1
    queued_state, _ = queue.pop()
    assert queued_state == root


def test_unselected_skeletons_are_failed_and_only_selected_children_activate():
    root = ProofState("", "root")
    chosen_child = ProofState("", "chosen")
    ignored_child_1 = ProofState("", "ignored1")
    ignored_child_2 = ProofState("", "ignored2")
    graph = ANDORGraph(root)
    chosen = Action("skeleton", "chosen", children=(chosen_child, ProofState("", "chosen2")))
    ignored_1 = Action("skeleton", "ignored1", children=(ignored_child_1,))
    ignored_2 = Action("skeleton", "ignored2", children=(ignored_child_2,))
    for action in (chosen, ignored_1, ignored_2):
        graph.expand(root, action, r_env=1.0)
    stats = {root: StateStats(depth=0)}
    queue = StatePriorityQueue()
    rollout = make_rollout(
        skeleton_beam_per_state=1,
        scorer=FixedScoreScorer({"chosen": 10.0, "ignored1": 1.0, "ignored2": 0.0}),
    )

    selected = rollout.select_skeletons(
        [(root, chosen), (root, ignored_1), (root, ignored_2)],
        graph,
        stats,
    )
    rollout.activate_skeleton_children(selected, graph, stats, queue, {rollout.state_key(root)})

    assert selected == [(root, chosen, 10.0)]
    assert graph.status(chosen) == "OPEN"
    assert graph.status(ignored_1) == "FAILED"
    assert graph.status(ignored_2) == "FAILED"
    assert {state.goal for _, state in queue.items()} == {"chosen", "chosen2"}


def test_activated_child_inherits_skeleton_score_and_r_env():
    root = ProofState("", "root")
    child = ProofState("", "child")
    graph = ANDORGraph(root)
    action = Action("skeleton", "split", children=(child,))
    graph.expand(root, action, r_env=0.5)
    stats = {root: StateStats(depth=0, last_score=1.0)}
    queue = StatePriorityQueue()
    rollout = make_rollout()
    expected_skeleton_score = rollout.scorer.score_skeleton(action, root, graph, stats)

    rollout.activate_skeleton_children(
        [(root, action, expected_skeleton_score)],
        graph,
        stats,
        queue,
        {rollout.state_key(root)},
    )

    assert stats[child].incoming_skeleton_score == expected_skeleton_score
    assert stats[child].incoming_skeleton_r_env == 0.5
    assert stats[child].last_score == rollout.scorer.score_state(child, graph, stats)


def test_propagate_solves_root_through_multiple_skeleton_layers():
    root = ProofState("", "root")
    child = ProofState("", "child")
    grandchild = ProofState("", "grandchild")
    graph = ANDORGraph(root)
    skel1 = Action("skeleton", "skel1", children=(child,))
    skel2 = Action("skeleton", "skel2", children=(grandchild,))
    tactic = Action("tactic", "solve", extracted_code="trivial")
    graph.expand(root, skel1, r_env=1.0)
    graph.expand(child, skel2, r_env=1.0)
    graph.expand(grandchild, tactic, r_env=1.0, tactic_status="SOLVED")
    stats = {
        root: StateStats(depth=0),
        child: StateStats(depth=1),
        grandchild: StateStats(depth=2),
    }
    rollout = make_rollout(max_depth=3)

    rollout.propagate(graph, stats)

    assert graph.status(grandchild) == "SOLVED"
    assert graph.status(skel2) == "SOLVED"
    assert graph.status(child) == "SOLVED"
    assert graph.status(skel1) == "SOLVED"
    assert graph.status(root) == "SOLVED"


def test_failed_skeleton_keeps_own_score_but_does_not_backup_to_parent_value():
    root = ProofState("", "root")
    child = ProofState("", "child")
    graph = ANDORGraph(root)
    skeleton = Action("skeleton", "split", children=(child,))
    graph.expand(root, skeleton, r_env=1.0)
    graph.mark_failed(child)
    rollout = make_rollout()
    stats = {root: StateStats(depth=0), child: StateStats(depth=1, exhausted=True)}

    rollout.propagate(graph, stats)
    q_values = graph.backup()

    assert graph.status(skeleton) == "FAILED"
    assert q_values[skeleton] == 1.0
    assert graph.backup_value_for_action(skeleton, q_values[skeleton]) == 0.0
    assert max(
        (graph.backup_value_for_action(a, q_values.get(a, 0.0)) for a in graph.get_actions(root)),
        default=0.0,
    ) == 0.0


def test_state_score_prioritizes_last_open_child_of_parent_skeleton():
    root = ProofState("", "root")
    solved_child_1 = ProofState("", "solved1")
    solved_child_2 = ProofState("", "solved2")
    target = ProofState("", "target")
    unrelated = ProofState("", "unrelated")
    graph = ANDORGraph(root)
    skeleton = Action("skeleton", "split", children=(solved_child_1, solved_child_2, target))
    graph.expand(root, skeleton, r_env=1.0)
    graph.add_state(solved_child_1, depth=1)
    graph.add_state(solved_child_2, depth=1)
    graph.add_state(target, depth=1)
    graph.add_state(unrelated, depth=1)
    graph.mark_solved(solved_child_1)
    graph.mark_solved(solved_child_2)
    stats = {
        root: StateStats(depth=0),
        solved_child_1: StateStats(depth=1),
        solved_child_2: StateStats(depth=1),
        target: StateStats(depth=1, parent_skeletons=[(root, skeleton)]),
        unrelated: StateStats(depth=1),
    }
    scorer = SimpleHeuristicScorer()

    target_score = scorer.score_state(target, graph, stats)
    unrelated_score = scorer.score_state(unrelated, graph, stats)

    assert target_score > unrelated_score + 7.0


def test_finalize_unresolved_keeps_open_states_and_actions_open():
    root = ProofState("", "root")
    child = ProofState("", "child")
    graph = ANDORGraph(root)
    skeleton = Action("skeleton", "split", children=(child,))
    graph.expand(root, skeleton, r_env=1.0)
    graph.add_state(child, depth=1)
    stats = {
        root: StateStats(depth=0),
        child: StateStats(depth=1),
    }
    rollout = make_rollout()

    rollout.finalize_unresolved(graph, stats)
    rollout.propagate(graph, stats)

    assert stats[root].exhausted
    assert stats[child].exhausted
    assert graph.status(root) == "OPEN"
    assert graph.status(skeleton) == "OPEN"
    assert graph.status(child) == "OPEN"


def test_search_metadata_logs_core_runtime_counters():
    root = ProofState("", "root")
    child = ProofState("", "child")
    executor = ScriptedExecutor(
        tactic_status=lambda state: "SOLVED" if state == child else "FAILED",
        skeleton_children=lambda state: (child, child) if state == root else (),
    )
    rollout = make_rollout(
        executor=executor,
        max_depth=2,
        max_nodes=10,
        initial_tactic_k=1,
        initial_skeleton_k=1,
    )

    _, graph, _ = rollout.rollout(root)
    meta = graph.search_metadata

    assert set(meta) == {
        "budget",
        "skeleton_pipeline",
        "final_status",
        "depth_distribution",
        "beam",
    }
    assert meta["budget"]["used_total"] <= meta["budget"]["max_nodes"]
    assert meta["budget"]["used_tactic"] >= 2
    assert meta["budget"]["used_skeleton_raw"] == 1
    assert meta["skeleton_pipeline"]["requested"] == 1
    assert meta["skeleton_pipeline"]["inserted_raw"] == 1
    assert meta["skeleton_pipeline"]["selected_by_beam"] == 1
    assert meta["skeleton_pipeline"]["children_new"] == 1
    assert meta["skeleton_pipeline"]["children_duplicate"] == 1
    assert meta["final_status"]["states"]["SOLVED"] >= 2
    assert meta["final_status"]["actions"]["skeleton_SOLVED"] == 1
    assert meta["depth_distribution"]["states_seen_by_depth"]["0"] == 1
