from __future__ import annotations

from gammazero.core import Action, ProofState
from gammazero.search.graph import ANDORGraph
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
    counts = rollout.run_jobs(graph, jobs, "tactic")
    for state, count in counts.items():
        stats[state].tactic_tries += count

    assert jobs == [(root, 2), (other, 2)]
    assert len(graph.get_actions(root)) + len(graph.get_actions(other)) == 2
    assert stats[root].tactic_tries + stats[other].tactic_tries == 2
    assert max(stats[root].tactic_tries, stats[other].tactic_tries) <= 2


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
        score = rollout.H_state(root, graph, stats)
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
