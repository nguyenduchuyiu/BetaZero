from __future__ import annotations

from gammazero.core import Action, ProofState
from gammazero.env.lean_env import LeanEnv
from gammazero.policy.output_parser import INVALID_SKELETON_FEEDBACK, TRUNCATED_THINK_FEEDBACK
from gammazero.policy.prompt import build_skeleton_retry_prompt, build_tactic_retry_prompt
from gammazero.search.graph import ANDORGraph
from gammazero.search.rollout.heuristic import SimpleHeuristicScorer
from gammazero.search.rollout.best_first_rollout import BestFirstRollout
from gammazero.search.rollout.batch_executor import BatchExecutor, RolloutBudget
from gammazero.search.rollout.execution_result import LeanExecutionResult
from gammazero.search.rollout.search_queue import StatePriorityQueue
from gammazero.search.rollout.search_stats import StateStats
from gammazero.utils.graph_logger import GraphLogger
from gammazero.utils.scaffold import target_subgoal_label


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


class FakeScheduler:
    executor = None


class BadFinalGoalSkeletonLean:
    scheduler = FakeScheduler()

    def execute(self, state, code):
        return (
            "theorem my_theorem : True := by\n  sorry",
            {"pass": True, "complete": False, "errors": [], "warnings": [], "sorries": [{}]},
            [ProofState(state.context, state.goal, state.header)],
        )


class NoopFailure:
    def handle_system_execute_failure(self, *args, **kwargs):
        raise AssertionError("unexpected system failure")

    def compute_failed_action_patch(self, *args, **kwargs):
        raise AssertionError("policy rejection should not run patching")

    def apply_failed_action_patch(self, *args, **kwargs):
        raise AssertionError("policy rejection should not apply patch")


class RecordingSystemFailure:
    def __init__(self):
        self.results: list[LeanExecutionResult] = []

    def handle_system_execute_failure(
        self,
        graph,
        state,
        action_kind,
        action_content,
        result,
        prompt="",
    ):
        self.results.append(result)
        graph.expand(
            state,
            Action(
                action_type=action_kind,
                content=action_content,
                extracted_code="",
                prompt=prompt,
                verify_code=result.state_code,
                lean_feedback=result.system_errors,
            ),
            r_env=0.0,
            tactic_status="FAILED" if action_kind == "tactic" else None,
        )

    def compute_failed_action_patch(self, *args, **kwargs):
        raise AssertionError("system extraction failure should not run patching")

    def apply_failed_action_patch(self, *args, **kwargs):
        raise AssertionError("system extraction failure should not apply patch")


class FixedScoreScorer:
    def __init__(self, skeleton_scores):
        self.skeleton_scores = skeleton_scores

    def score_state(self, state, graph, stats):
        return 0.0

    def score_skeleton(self, action, parent_state, graph, stats):
        return self.skeleton_scores[action.content]


class FakeLeanNoSubgoals:
    scheduler = None

    def execute(self, state, code):
        return (
            "theorem my_theorem : True := by\n  sorry",
            {"pass": True, "complete": False, "sorries": [], "warnings": []},
            [],
        )


class SubgoalTacticLean:
    scheduler = FakeScheduler()

    def __init__(self, *, pass_result=True):
        self.pass_result = pass_result
        self.execute_calls = 0
        self.verify_calls: list[str] = []

    def execute(self, state, code):
        self.execute_calls += 1
        raise AssertionError("subgoal child tactics must not use isolated execute")

    def verify(self, code):
        self.verify_calls.append(code)
        if self.pass_result:
            return {
                "pass": True,
                "complete": False,
                "errors": [],
                "warnings": [{"severity": "warning", "data": "declaration uses 'sorry'"}],
                "sorries": [{"goal": "sibling admit"}],
            }
        return {
            "pass": False,
            "complete": False,
            "errors": [{"severity": "error", "data": "subgoal failure"}],
            "warnings": [],
            "sorries": [],
        }


class PatchableSubgoalTacticLean:
    scheduler = FakeScheduler()

    def __init__(self):
        self.execute_calls = 0
        self.verify_calls: list[str] = []

    def execute(self, state, code):
        self.execute_calls += 1
        raise AssertionError("subgoal child tactics must not use isolated execute")

    def verify(self, code):
        self.verify_calls.append(code)
        if "exact fixed" in code:
            return {
                "pass": True,
                "complete": False,
                "errors": [],
                "warnings": [{"severity": "warning", "data": "declaration uses 'sorry'"}],
                "sorries": [{"goal": "sibling admit"}],
            }
        return {
            "pass": False,
            "complete": False,
            "errors": [{"severity": "error", "data": "subgoal failure"}],
            "warnings": [],
            "sorries": [],
        }

    def analyze_dependencies(self, proof_code, allowed_vars=None, target_name=None):
        return {
            "core_solved": ["MAIN_GOAL"],
            "core_failed": [],
            "benign": [],
            "malignant": [],
        }


class FixBadSorrifier:
    def fix_code(self, code):
        return code.replace("exact bad", "exact fixed")


class SubgoalPatchFailure:
    def handle_system_execute_failure(self, *args, **kwargs):
        raise AssertionError("unexpected system failure")

    def compute_failed_action_patch(self, *args, **kwargs):
        raise AssertionError("subgoal tactic should use the subgoal patch path")

    def _new_sorrifier(self):
        return FixBadSorrifier()

    def apply_failed_action_patch(self, graph, patch):
        graph.expand(
            patch.state,
            Action(
                action_type=patch.action_kind,
                content=patch.action_content,
                extracted_code=patch.lean_code,
                children=(),
                prompt=patch.prompt,
            ),
            r_env=patch.r_fail,
            r_dep=patch.r_dep,
            tactic_status="FAILED",
        )
        return patch.patched_action_code


class FixedReward:
    def __init__(self, r_env):
        self.r_env_value = r_env
        self.calls: list[tuple[str, str, dict]] = []

    def r_env(self, full_orig, full_patched, patched_vr):
        self.calls.append((full_orig, full_patched, patched_vr))
        return self.r_env_value

    def r_dep(self, dep_graph):
        n_c = len(dep_graph.get("core", []))
        n_b = len(dep_graph.get("benign", []))
        n_m = len(dep_graph.get("malignant", []))
        if n_c == 0:
            return 0.0
        return n_c / (n_c + 0.5 * n_b + 2.0 * n_m)


class FixedSubgoalLean:
    def __init__(self, result):
        self.result = result

    def execute(self, state, code):
        return self.result

    def verify(self, code):
        raise AssertionError("unexpected verify call")


class StaticVerifyScheduler:
    def __init__(self, verify_result):
        self.verify_result = verify_result
        self.verify_calls: list[str] = []

    def verify(self, code):
        self.verify_calls.append(code)
        return self.verify_result


class NoopFailureHandler:
    def handle_system_execute_failure(self, *args, **kwargs):
        pass

    def compute_failed_action_patch(self, *args, **kwargs):
        raise AssertionError("zero-subgoal pass should not enter patch path")

    def apply_failed_action_patch(self, *args, **kwargs):
        raise AssertionError("zero-subgoal pass should not enter patch path")


def make_rollout(policy=None, executor=None, **kwargs):
    defaults = dict(
        max_depth=1,
        max_nodes=32,
        initial_tactic_k=1,
        retry_tactic_k=1,
        max_tactic_per_state=1,
        min_tactic_before_skeleton=1,
        promising_tactic_r_env=2.0,
        strong_tactic_r_env=2.0,
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


def test_duplicate_only_skeleton_round_links_active_children_without_new_nodes():
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
    rollout.update_skeleton_progress(
        [(root, action, 0.0)],
        child_stats["generated_by_parent"],
        stats,
        child_stats["active_by_parent"],
    )

    assert child_stats["generated"] == 0
    assert child_stats["duplicates"] == 2
    assert stats[root].last_skeleton_new_children == 0
    assert stats[root].active_skeleton_children == {duplicate}
    assert stats[root].bad_skeleton_rounds == 0
    assert not stats[root].skeleton_exhausted
    assert rollout.should_expand_skeleton(root, graph, stats)
    assert rollout.should_try_tactic(root, graph, stats)


def test_reserved_skeleton_blocks_new_skeleton_sampling():
    root = ProofState("", "root")
    child = ProofState("", "child")
    graph = ANDORGraph(root)
    reserved = Action("skeleton", "reserved", children=(child,))
    graph.expand(root, reserved, r_env=1.0)
    graph.mark_failed(reserved)
    stats = {
        root: StateStats(
            depth=0,
            tactic_tries=1,
            skeleton_tries=4,
            tactic_probe_done=True,
            reserved_skeletons=[(1.0, reserved)],
        )
    }
    rollout = make_rollout(max_tactic_per_state=8, max_skeleton_per_state=12)

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


def test_skeleton_waits_for_min_tactic_probe_budget():
    root = ProofState("", "root")
    graph = ANDORGraph(root)
    stats = {
        root: StateStats(
            depth=0,
            tactic_tries=4,
            tactic_probe_done=True,
            best_tactic_r_env=0.0,
        )
    }
    rollout = make_rollout(
        max_tactic_per_state=12,
        max_skeleton_per_state=12,
        min_tactic_before_skeleton=8,
    )

    assert not rollout.should_expand_skeleton(root, graph, stats)
    stats[root].tactic_tries = 8
    assert rollout.should_expand_skeleton(root, graph, stats)


def test_promising_tactic_signal_delays_skeleton_until_tactic_quota():
    root = ProofState("", "root")
    graph = ANDORGraph(root)
    stats = {
        root: StateStats(
            depth=0,
            tactic_tries=8,
            tactic_probe_done=True,
            best_tactic_r_env=0.5,
        )
    }
    rollout = make_rollout(
        max_tactic_per_state=12,
        max_skeleton_per_state=12,
        min_tactic_before_skeleton=8,
        promising_tactic_r_env=0.4,
        strong_tactic_r_env=0.7,
    )

    assert not rollout.should_expand_skeleton(root, graph, stats)
    stats[root].tactic_tries = 12
    assert rollout.should_expand_skeleton(root, graph, stats)


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


def test_duplicate_skeleton_decomposition_is_rejected():
    root = ProofState("", "root")
    child1 = ProofState("", "child1")
    child2 = ProofState("", "child2")
    graph = ANDORGraph(root)
    first = Action("skeleton", "split with h_log_add", children=(child1, child2))
    duplicate = Action("skeleton", "split with h_log_mul", children=(child1, child2))
    graph.expand(root, first, r_env=1.0)
    graph.expand(root, duplicate, r_env=1.0)
    rollout = make_rollout()

    valid = rollout.valid_skeletons_from_actions(graph, [first, duplicate])

    assert valid == [(root, first)]
    assert graph.status(first) == "OPEN"
    assert graph.status(duplicate) == "FAILED"
    assert rollout._duplicate_skeleton_actions == 1


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
    assert "FAILED CHECKED CODE" in prompt
    assert "unknown identifier 'bad'" in prompt
    assert "Do not repeat the failed skeleton" in prompt
    assert prompt.index("[PROBLEM]") < prompt.index("PREVIOUS SKELETON ATTEMPTS FAILED")


def test_bad_final_goal_skeleton_is_failed_with_policy_feedback():
    root = ProofState("h : True", "False")
    graph = ANDORGraph(root)
    executor = BatchExecutor(
        BadFinalGoalSkeletonLean(),
        NoopFailure(),
        FakeReward(),
        max_workers=1,
    )
    raw = (
        "```lean4\n"
        "theorem my_theorem (h : True) : False := by\n"
        "  have h_final : False := sorry\n"
        "  exact h_final\n"
        "```"
    )

    feedbacks = executor.execute(
        graph,
        [root],
        [[{"text": raw}]],
        "skeleton",
        RolloutBudget(4),
        prompts=["prompt"],
    )
    actions = graph.get_actions(root)

    assert len(actions) == 1
    assert graph.status(actions[0]) == "FAILED"
    assert graph.get_r_env(actions[0]) == 0.0
    assert "SKELETON POLICY VIOLATION" in feedbacks[0][0][1]
    assert "BAD EXAMPLE" in feedbacks[0][0][1]


def test_subgoal_skeleton_with_admit_and_naked_sorry_is_policy_failed_before_verify():
    root = ProofState("Child Sibling Part : Prop", "True")
    child = ProofState("Child Sibling Part : Prop", "Child")
    sibling = ProofState("Child Sibling Part : Prop", "Sibling")
    skeleton = Action(
        "skeleton",
        "skel",
        extracted_code="have h_child : Child := sorry\nhave h_sibling : Sibling := sorry\ntrivial",
        children=(child, sibling),
    )
    graph = ANDORGraph(root)
    graph.expand(root, skeleton, r_env=1.0)
    graph.add_state(child, depth=1)
    lean = SubgoalTacticLean(pass_result=True)
    executor = BatchExecutor(lean, NoopFailure(), FakeReward(), max_workers=1)
    raw = (
        "```lean4\n"
        "theorem my_theorem (Child Sibling Part : Prop) : True := by\n"
        "  have h_child : Child := by\n"
        "    have h_part : Part := sorry\n"
        "    admit\n"
        "    sorry\n"
        "  have h_sibling : Sibling := by admit\n"
        "  trivial\n"
        "```"
    )

    feedbacks = executor.execute(
        graph,
        [child],
        [[{"text": raw}]],
        "skeleton",
        RolloutBudget(4),
        prompts=["subgoal skeleton prompt"],
    )

    actions = graph.get_actions(child)
    assert lean.verify_calls == []
    assert len(actions) == 1
    assert graph.status(actions[0]) == "FAILED"
    assert actions[0].extracted_code == "have h_part : Part := sorry\nadmit\nsorry"
    assert actions[0].lean_feedback == INVALID_SKELETON_FEEDBACK
    assert feedbacks[0][0][1] == INVALID_SKELETON_FEEDBACK


def test_empty_extraction_records_failed_action_without_retry_feedback():
    root = ProofState("", "True")
    graph = ANDORGraph(root)
    failure = RecordingSystemFailure()
    lean = SubgoalTacticLean(pass_result=True)
    executor = BatchExecutor(lean, failure, FakeReward(), max_workers=1)

    feedbacks = executor.execute(
        graph,
        [root],
        [[{"text": "<think>no final lean block</think>"}]],
        "tactic",
        RolloutBudget(4),
        prompts=["prompt"],
    )

    actions = graph.get_actions(root)
    assert feedbacks == [[None]]
    assert len(failure.results) == 1
    assert failure.results[0].system_errors == TRUNCATED_THINK_FEEDBACK
    assert len(actions) == 1
    assert graph.status(actions[0]) == "FAILED"
    assert actions[0].extracted_code == ""


def test_tactic_feedback_is_cached_and_added_to_retry_prompt():
    root = ProofState("", "root")
    graph = ANDORGraph(root)
    rollout = make_rollout(executor=FeedbackExecutor())

    rollout.run_jobs(graph, [(root, 1)], "tactic")
    prompt = build_tactic_retry_prompt(root, rollout._tactic_feedback_by_state[root])
    built_prompt = rollout.prompt_builder.build(
        root,
        "tactic",
        tactic_feedbacks=rollout._tactic_feedback_by_state[root],
    )

    assert "PREVIOUS TACTIC ATTEMPTS FAILED" in prompt
    assert "FAILED CHECKED CODE" in prompt
    assert "unknown identifier 'bad'" in prompt
    assert "Do not repeat the failed tactic" in prompt
    assert "FAILED CHECKED CODE" in built_prompt
    assert prompt.index("[PROBLEM]") < prompt.index("PREVIOUS TACTIC ATTEMPTS FAILED")
    assert built_prompt.index("[PROBLEM]") < built_prompt.index("PREVIOUS TACTIC ATTEMPTS FAILED")


def test_subgoal_child_tactic_is_verified_inside_parent_skeleton():
    root = ProofState("h : True\nChild : Prop\nSibling : Prop", "True")
    child = ProofState("h : True\nChild : Prop\nSibling : Prop", "Child")
    sibling = ProofState("h : True\nChild : Prop\nSibling : Prop", "Sibling")
    skeleton = Action(
        "skeleton",
        "skel",
        extracted_code="have h_child : Child := sorry\nhave h_sibling : Sibling := sorry\ntrivial",
        children=(child, sibling),
    )
    graph = ANDORGraph(root)
    graph.expand(root, skeleton, r_env=1.0)
    graph.add_state(child, depth=1)
    lean = SubgoalTacticLean(pass_result=True)
    executor = BatchExecutor(lean, NoopFailure(), FakeReward(), max_workers=1)
    raw = (
        "<think>\nUse the available proof of True for the target subgoal.\n</think>\n"
        "```lean4\n"
        "theorem my_theorem (h : True) (Child Sibling : Prop) : True := by\n"
        "  have h_child : Child := by\n"
        "    exact h\n"
        "  have h_sibling : Sibling := admit\n"
        "  trivial\n"
        "```"
    )

    feedbacks = executor.execute(
        graph,
        [child],
        [[{"text": raw}]],
        "tactic",
        RolloutBudget(4),
        prompts=["subgoal prompt"],
    )

    actions = graph.get_actions(child)
    assert feedbacks == [[None]]
    assert lean.execute_calls == 0
    assert len(lean.verify_calls) == 1
    assert "have h_child : Child := by\n    exact h" in lean.verify_calls[0]
    assert "have h_sibling : Sibling := by\n    admit" in lean.verify_calls[0]
    assert graph.status(actions[0]) == "SOLVED"


def test_subgoal_tactic_rejects_forbidden_placeholders_before_verify():
    root = ProofState("h : True\nChild : Prop", "True")
    child = ProofState("h : True\nChild : Prop", "Child")
    skeleton = Action(
        "skeleton",
        "skel",
        extracted_code="have h_child : Child := sorry\ntrivial",
        children=(child,),
    )
    graph = ANDORGraph(root)
    graph.expand(root, skeleton, r_env=1.0)
    graph.add_state(child, depth=1)
    lean = SubgoalTacticLean(pass_result=True)
    executor = BatchExecutor(lean, NoopFailure(), FakeReward(), max_workers=1)
    raw = (
        "<think>\nThis attempt is bad because it uses a placeholder.\n</think>\n"
        "```lean4\n"
        "theorem my_theorem (h : True) (Child : Prop) : True := by\n"
        "  have h_child : Child := by\n"
        "    -- comment may mention sorry\n"
        "    exact admit\n"
        "  trivial\n"
        "```"
    )

    feedbacks = executor.execute(
        graph,
        [child],
        [[{"text": raw}]],
        "tactic",
        RolloutBudget(4),
        prompts=["subgoal prompt"],
    )

    actions = graph.get_actions(child)
    assert lean.verify_calls == []
    assert graph.status(actions[0]) == "FAILED"
    assert "TACTIC POLICY VIOLATION" in feedbacks[0][0][1]


def test_failed_subgoal_tactic_scores_only_marked_target_lines_in_parent_context():
    root = ProofState("h : True\nChild : Prop\nSibling : Prop", "True")
    child = ProofState("h : True\nChild : Prop\nSibling : Prop", "Child")
    sibling = ProofState("h : True\nChild : Prop\nSibling : Prop", "Sibling")
    skeleton = Action(
        "skeleton",
        "skel",
        extracted_code="have h_child : Child := sorry\nhave h_sibling : Sibling := sorry\ntrivial",
        children=(child, sibling),
    )
    graph = ANDORGraph(root)
    graph.expand(root, skeleton, r_env=1.0)
    graph.add_state(child, depth=1)
    lean = PatchableSubgoalTacticLean()
    reward = FixedReward(0.42)
    executor = BatchExecutor(lean, SubgoalPatchFailure(), reward, max_workers=1)
    raw = (
        "<think>\nTry a tactic for the target subgoal.\n</think>\n"
        "```lean4\n"
        "theorem my_theorem (h : True) (Child Sibling : Prop) : True := by\n"
        "  have h_child : Child := by\n"
        "    exact bad\n"
        "  have h_sibling : Sibling := by admit\n"
        "  trivial\n"
        "```"
    )

    feedbacks = executor.execute(
        graph,
        [child],
        [[{"text": raw}]],
        "tactic",
        RolloutBudget(4),
        prompts=["subgoal prompt"],
    )

    actions = graph.get_actions(child)
    assert len(actions) == 1
    assert lean.execute_calls == 0
    assert len(lean.verify_calls) == 2
    assert graph.status(actions[0]) == "FAILED"
    assert graph.get_r_env(actions[0]) == 0.42
    assert graph._r_dep[actions[0]] == 1.0
    assert feedbacks[0][0][2] == "exact fixed"
    full_orig, full_patched, _ = reward.calls[0]
    assert "exact bad" in full_orig
    assert "exact fixed" in full_patched
    assert "theorem my_theorem (h : True) (Child : Prop) (Sibling : Prop) : True" in full_orig
    assert "theorem my_theorem (h : True) (Child : Prop) (Sibling : Prop) : True" in full_patched
    assert "h_sibling" not in full_orig
    assert "h_sibling" not in full_patched
    assert "GAMMAZERO_TARGET_SCORE" not in full_orig
    assert "GAMMAZERO_TARGET_SCORE" not in full_patched


def test_failed_tactic_backup_uses_r_env_and_r_dep():
    root = ProofState("", "True")
    graph = ANDORGraph(root)
    tactic = Action("tactic", "bad", extracted_code="bad")
    graph.expand(root, tactic, r_env=0.25, r_dep=1.0, tactic_status="FAILED")

    q_values = graph.backup()

    assert graph.status(tactic) == "FAILED"
    assert q_values[tactic] == 1.25
    assert graph.backup_value_for_action(tactic, q_values[tactic]) == 1.25


def test_subgoal_tactic_prompt_keeps_only_target_sorry():
    root = ProofState("h : True", "True")
    child = ProofState("h : True\nChild : Prop", "Child")
    sibling = ProofState("h : True\nSibling : Prop", "Sibling")
    graph = ANDORGraph(root)
    skeleton = Action(
        "skeleton",
        "skel",
        extracted_code="have h_child : Child := sorry\nhave h_sibling : Sibling := sorry\ntrivial",
        children=(child, sibling),
    )
    graph.expand(root, skeleton, r_env=1.0)
    graph.add_state(child, depth=1)
    rollout = make_rollout()

    prompts = [
        rollout.prompt_builder.build_subgoal_tactic(root, skeleton, 0)
    ]

    problem = prompts[0].split("[PROBLEM]", 1)[1]
    code = problem.rsplit("```lean4", 1)[1].split("```", 1)[0]
    assert code.count("sorry") == 1
    assert code.count("admit") >= 1
    assert "have h_sibling : Sibling := by admit" in code
    assert "whole parent theorem scaffold" in prompts[0]


def test_lean_execute_child_scaffolds_isolate_sibling_sorries():
    parent_scaffold = (
        "theorem my_theorem (Child Sibling : Prop) : True := by\n"
        "  sorry\n"
    )
    state = ProofState(
        "Child Sibling : Prop",
        "True",
        scaffold_code=parent_scaffold,
        target_index=0,
        target_kind="root",
    )
    scheduler = StaticVerifyScheduler(
        {
            "pass": True,
            "complete": False,
            "errors": [],
            "warnings": [],
            "sorries": [{"goal": "⊢ Child"}, {"goal": "⊢ Sibling"}],
        }
    )
    lean = LeanEnv(scheduler)

    _, _, children = lean.execute(
        state,
        "have h_child : Child := sorry\nhave h_sibling : Sibling := sorry\ntrivial",
    )

    assert len(children) == 2
    assert children[0].target_index == 0
    assert children[0].scaffold_code.count("sorry") == 1
    assert children[0].scaffold_code.count("admit") == 1
    assert "have h_child : Child := sorry" in children[0].scaffold_code
    assert "have h_sibling : Sibling := by\n    admit" in children[0].scaffold_code
    assert children[1].target_index == 0
    assert children[1].scaffold_code.count("sorry") == 1
    assert children[1].scaffold_code.count("admit") == 1
    assert "have h_child : Child := by\n    admit" in children[1].scaffold_code
    assert "have h_sibling : Sibling := sorry" in children[1].scaffold_code


def test_subgoal_prompt_uses_child_scaffold_with_nested_siblings_admitted():
    child_scaffold = (
        "theorem my_theorem (H Part Other Sibling : Prop) : True := by\n"
        "  have h_equiv : H := by\n"
        "    have h_x_sol : Part := sorry\n"
        "    have h_other : Other := by\n"
        "      admit\n"
        "    admit\n"
        "  have h_sibling : Sibling := by\n"
        "    admit\n"
        "  trivial\n"
    )
    root = ProofState("H Sibling : Prop", "True")
    child = ProofState(
        "H Part Other Sibling : Prop",
        "Part",
        scaffold_code=child_scaffold,
        target_index=0,
        target_kind="mini_skeleton_child",
    )
    sibling = ProofState("H Part Other Sibling : Prop", "Other")
    skeleton = Action(
        "skeleton",
        "skel",
        extracted_code="have h_x_sol : Part := sorry\nhave h_other : Other := sorry\nadmit",
        children=(child, sibling),
    )
    rollout = make_rollout()

    prompt = rollout.prompt_builder.build_subgoal_tactic(
        root,
        skeleton,
        0,
        child_state=child,
    )

    problem = prompt.split("[PROBLEM]", 1)[1]
    code = problem.rsplit("```lean4", 1)[1].split("```", 1)[0]
    assert code.count("sorry") == 1
    assert "have h_x_sol : Part := sorry" in code
    assert "have h_other : Other := by\n      admit" in code
    assert "have h_sibling : Sibling := by\n    admit" in code


def test_scaffold_target_label_names_have_containing_sorry():
    scaffold = (
        "theorem my_theorem (Child Part : Prop) : True := by\n"
        "  have h_child : Child := by\n"
        "    have h_part : Part := sorry\n"
        "    admit\n"
        "  have h_done : True := sorry\n"
        "  trivial\n"
    )

    assert target_subgoal_label(scaffold, 0) == "h_part"
    assert target_subgoal_label(scaffold, 1) == "h_done"


def test_graph_logger_exports_or_node_target_label():
    scaffold = (
        "theorem my_theorem (Child : Prop) : True := by\n"
        "  have h_child : Child := sorry\n"
        "  trivial\n"
    )
    root = ProofState(
        "Child : Prop",
        "Child",
        scaffold_code=scaffold,
        target_index=0,
        target_kind="skeleton_child",
    )
    graph = ANDORGraph(root)

    data = GraphLogger().export_to_dict(graph, root, {})

    root_node = next(node for node in data["nodes"] if node["id"] == data["root_id"])
    assert root_node["target_label"] == "h_child"
    assert root_node["content"]["target_label"] == "h_child"


def test_graph_logger_exports_and_node_target_child_label():
    root = ProofState("Child Sibling : Prop", "True")
    child = ProofState("Child Sibling : Prop", "Child")
    sibling = ProofState("Child Sibling : Prop", "Sibling")
    graph = ANDORGraph(root)
    skeleton = Action(
        "skeleton",
        "skel",
        extracted_code="have h_child : Child := sorry\nhave h_sibling : Sibling := sorry\ntrivial",
        children=(child, sibling),
    )
    graph.expand(root, skeleton, r_env=1.0)
    tactic = Action(
        "tactic",
        "solve child",
        extracted_code="exact trivial",
        target_child_index=0,
    )
    graph.expand(child, tactic, r_env=1.0, tactic_status="SOLVED")

    data = GraphLogger().export_to_dict(graph, root, {})

    action_node = next(
        node
        for node in data["nodes"]
        if node["type"] == "AND" and node["content"] == "solve child"
    )
    assert action_node["target_label"] == "h_child"
    assert action_node["target_child_label"] == "h_child"


def test_subgoal_tactic_prompt_names_target_subgoal():
    root = ProofState("h : True", "True")
    child = ProofState("h : True\nChild : Prop", "Child")
    sibling = ProofState("h : True\nSibling : Prop", "Sibling")
    skeleton = Action(
        "skeleton",
        "skel",
        extracted_code="have h_child : Child := sorry\nhave h_sibling : Sibling := sorry\ntrivial",
        children=(child, sibling),
    )
    rollout = make_rollout()

    prompt = rollout.prompt_builder.build_subgoal_tactic(root, skeleton, 0)

    assert "[TARGET SUBGOAL]" in prompt
    assert "name: h_child" in prompt
    assert "child_index: 0" in prompt
    assert "goal: Child" in prompt
    assert "Solve exactly the `sorry` for target subgoal `h_child`" in prompt


def test_subgoal_skeleton_prompt_keeps_parent_scaffold_context():
    root = ProofState("h : True", "True")
    child = ProofState("h : True\nChild : Prop", "Child")
    sibling = ProofState("h : True\nSibling : Prop", "Sibling")
    graph = ANDORGraph(root)
    skeleton = Action(
        "skeleton",
        "skel",
        extracted_code=(
            "have h_pre : True := by trivial\n"
            "have h_child : Child := sorry\n"
            "have h_sibling : Sibling := sorry\n"
            "trivial"
        ),
        children=(child, sibling),
    )
    graph.expand(root, skeleton, r_env=1.0)
    graph.add_state(child, depth=1)
    rollout = make_rollout()

    prompt = rollout.prompt_builder.build_subgoal_skeleton(root, skeleton, 0)

    problem = prompt.split("[PROBLEM]", 1)[1]
    code = problem.rsplit("```lean4", 1)[1].split("```", 1)[0]
    assert code.count("sorry") == 1
    assert code.count("admit") >= 1
    assert "have h_pre : True := by trivial" in code
    assert "have h_sibling : Sibling := by admit" in code
    assert "Subgoal Skeleton Generator" in prompt


def test_subgoal_skeleton_prompt_names_target_subgoal():
    root = ProofState("h : True", "True")
    child = ProofState("h : True\nChild : Prop", "Child")
    sibling = ProofState("h : True\nSibling : Prop", "Sibling")
    skeleton = Action(
        "skeleton",
        "skel",
        extracted_code="have h_child : Child := sorry\nhave h_sibling : Sibling := sorry\ntrivial",
        children=(child, sibling),
    )
    rollout = make_rollout()

    prompt = rollout.prompt_builder.build_subgoal_skeleton(root, skeleton, 0)

    assert "[TARGET SUBGOAL]" in prompt
    assert "name: h_child" in prompt
    assert "child_index: 0" in prompt
    assert "goal: Child" in prompt
    assert "Decompose exactly the `sorry` for target subgoal `h_child`" in prompt


def test_subgoal_child_skeleton_is_verified_inside_parent_skeleton():
    root = ProofState("h : True\nChild : Prop\nSibling : Prop", "True")
    child = ProofState("h : True\nChild : Prop\nSibling : Prop\nPart : Prop", "Child")
    sibling = ProofState("h : True\nChild : Prop\nSibling : Prop", "Sibling")
    parent_skeleton = Action(
        "skeleton",
        "skel",
        extracted_code=(
            "have h_pre : True := by trivial\n"
            "have h_child : Child := sorry\n"
            "have h_sibling : Sibling := sorry\n"
            "trivial"
        ),
        children=(child, sibling),
    )
    graph = ANDORGraph(root)
    graph.expand(root, parent_skeleton, r_env=1.0)
    graph.add_state(child, depth=1)
    lean = SubgoalTacticLean(pass_result=True)
    lean.verify = lambda code: (
        lean.verify_calls.append(code)
        or {
            "pass": True,
            "complete": False,
            "errors": [],
            "warnings": [{"severity": "warning", "data": "declaration uses 'sorry'"}],
            "sorries": [{"goal": "⊢ Part"}, {"goal": "⊢ Sibling"}],
        }
    )
    reward = FixedReward(0.42)
    executor = BatchExecutor(lean, NoopFailure(), reward, max_workers=1)
    raw = (
        "<think>\nDecompose the target subgoal in the parent scaffold.\n</think>\n"
        "```lean4\n"
        "theorem my_theorem (h : True) (Child Sibling Part : Prop) : True := by\n"
        "  have h_pre : True := by trivial\n"
        "  have h_child : Child := by\n"
        "    have h_part : Part := sorry\n"
        "    exact h_part\n"
        "  have h_sibling : Sibling := by admit\n"
        "  trivial\n"
        "```"
    )

    feedbacks = executor.execute(
        graph,
        [child],
        [[{"text": raw}]],
        "skeleton",
        RolloutBudget(4),
        prompts=["subgoal skeleton prompt"],
    )

    actions = graph.get_actions(child)
    assert feedbacks == [[None]]
    assert lean.execute_calls == 0
    assert len(lean.verify_calls) == 1
    assert "have h_pre : True := by trivial" in lean.verify_calls[0]
    assert "have h_child : Child := by\n    have h_part : Part := sorry\n    exact h_part" in lean.verify_calls[0]
    assert "have h_sibling : Sibling := by\n    admit" in lean.verify_calls[0]
    assert len(actions) == 1
    assert actions[0].action_type == "skeleton"
    assert actions[0].extracted_code == "have h_part : Part := sorry\nexact h_part"
    assert [s.goal for s in actions[0].children] == ["Part"]
    assert actions[0].children[0].target_index == 0
    assert actions[0].children[0].scaffold_code.count("sorry") == 1
    assert actions[0].children[0].scaffold_code.count("admit") == 1
    assert "have h_part : Part := sorry" in actions[0].children[0].scaffold_code
    assert "have h_sibling : Sibling := by\n    admit" in actions[0].children[0].scaffold_code
    assert graph.get_r_env(actions[0]) == 0.42


def test_failed_subgoal_child_skeleton_patch_scores_target_slice():
    root = ProofState("h : True\nChild : Prop\nSibling : Prop", "True")
    child = ProofState("h : True\nChild : Prop\nSibling : Prop\nPart : Prop", "Child")
    sibling = ProofState("h : True\nChild : Prop\nSibling : Prop", "Sibling")
    parent_skeleton = Action(
        "skeleton",
        "skel",
        extracted_code=(
            "have h_pre : True := by trivial\n"
            "have h_child : Child := sorry\n"
            "have h_sibling : Sibling := sorry\n"
            "trivial"
        ),
        children=(child, sibling),
    )
    graph = ANDORGraph(root)
    graph.expand(root, parent_skeleton, r_env=1.0)
    graph.add_state(child, depth=1)
    lean = PatchableSubgoalTacticLean()
    reward = FixedReward(0.37)
    executor = BatchExecutor(lean, SubgoalPatchFailure(), reward, max_workers=1)
    raw = (
        "<think>\nBad mini-skeleton first.\n</think>\n"
        "```lean4\n"
        "theorem my_theorem (h : True) (Child Sibling Part : Prop) : True := by\n"
        "  have h_pre : True := by trivial\n"
        "  have h_child : Child := by\n"
        "    exact bad\n"
        "  have h_sibling : Sibling := by admit\n"
        "  trivial\n"
        "```"
    )

    feedbacks = executor.execute(
        graph,
        [child],
        [[{"text": raw}]],
        "skeleton",
        RolloutBudget(4),
        prompts=["subgoal skeleton prompt"],
    )

    actions = graph.get_actions(child)
    assert len(actions) == 1
    assert graph.status(actions[0]) == "FAILED"
    assert graph.get_r_env(actions[0]) == 0.37
    assert feedbacks[0][0][2] == "exact fixed"
    full_orig, full_patched, _ = reward.calls[0]
    assert "exact bad" in full_orig
    assert "exact fixed" in full_patched
    assert "h_sibling" not in full_orig
    assert "h_sibling" not in full_patched


def test_subgoal_child_skeleton_targets_offset_sorry_index():
    root = ProofState("h : True\nChild : Prop\nSibling : Prop", "True")
    child = ProofState("h : True\nChild : Prop\nSibling : Prop", "Child")
    sibling = ProofState("h : True\nChild : Prop\nSibling : Prop\nPart : Prop", "Sibling")
    parent_skeleton = Action(
        "skeleton",
        "skel",
        extracted_code=(
            "have h_pre : True := by trivial\n"
            "have h_child : Child := sorry\n"
            "have h_sibling : Sibling := sorry\n"
            "trivial"
        ),
        children=(child, sibling),
    )
    graph = ANDORGraph(root)
    graph.expand(root, parent_skeleton, r_env=1.0)
    graph.add_state(sibling, depth=1)
    lean = SubgoalTacticLean(pass_result=True)
    lean.verify = lambda code: (
        lean.verify_calls.append(code)
        or {
            "pass": True,
            "complete": False,
            "errors": [],
            "warnings": [{"severity": "warning", "data": "declaration uses 'sorry'"}],
            "sorries": [{"goal": "⊢ Child"}, {"goal": "⊢ Part"}],
        }
    )
    reward = FixedReward(0.42)
    executor = BatchExecutor(lean, NoopFailure(), reward, max_workers=1)
    raw = (
        "<think>\nDecompose the target subgoal in the parent scaffold.\n</think>\n"
        "```lean4\n"
        "theorem my_theorem (h : True) (Child Sibling Part : Prop) : True := by\n"
        "  have h_pre : True := by trivial\n"
        "  have h_child : Child := by admit\n"
        "  have h_sibling : Sibling := by\n"
        "    have h_part : Part := sorry\n"
        "    exact h_part\n"
        "  trivial\n"
        "```"
    )

    executor.execute(
        graph,
        [sibling],
        [[{"text": raw}]],
        "skeleton",
        RolloutBudget(4),
        prompts=["subgoal skeleton prompt"],
    )

    assert len(lean.verify_calls) == 1
    assert "have h_child : Child := by\n    admit" in lean.verify_calls[0]
    assert "have h_sibling : Sibling := by\n    have h_part : Part := sorry" in lean.verify_calls[0]
    children = graph.get_actions(sibling)[0].children
    assert [s.goal for s in children] == ["Part"]
    assert children[0].target_index == 0
    assert children[0].scaffold_code.count("sorry") == 1
    assert children[0].scaffold_code.count("admit") == 1
    assert "have h_child : Child := by\n    admit" in children[0].scaffold_code
    assert "have h_part : Part := sorry" in children[0].scaffold_code


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


def test_unselected_skeletons_are_reserved_and_only_committed_children_activate():
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
    assert stats[root].committed_skeleton == chosen
    assert graph.status(chosen) == "OPEN"
    assert graph.status(ignored_1) == "FAILED"
    assert graph.status(ignored_2) == "FAILED"
    assert [action for _, action in stats[root].reserved_skeletons] == [ignored_1, ignored_2]
    assert {state.goal for _, state in queue.items()} == {"chosen", "chosen2"}


def test_failed_committed_skeleton_activates_best_reserved_fallback():
    root = ProofState("", "root")
    failed_child = ProofState("", "failed_child")
    fallback_child = ProofState("", "fallback_child")
    graph = ANDORGraph(root)
    committed = Action("skeleton", "committed", children=(failed_child,))
    fallback = Action("skeleton", "fallback", children=(fallback_child,))
    graph.expand(root, committed, r_env=1.0)
    graph.expand(root, fallback, r_env=1.0)
    graph.mark_failed(committed)
    stats = {
        root: StateStats(
            depth=0,
            committed_skeleton=committed,
            reserved_skeletons=[(5.0, fallback)],
        )
    }
    queue = StatePriorityQueue()
    rollout = make_rollout()
    seen = {rollout.state_key(root): root}

    result = rollout.activate_reserved_skeletons_for_failed_commits(
        [root],
        graph,
        stats,
        queue,
        seen,
    )

    assert result["fallback_activated"] == 1
    assert stats[root].committed_skeleton == fallback
    assert stats[root].skeleton_commit_failed_count == 1
    assert {state.goal for _, state in queue.items()} == {"fallback_child"}


def test_stale_committed_skeleton_keeps_commitment_and_does_not_parallel_fallback():
    root = ProofState("", "root")
    hard_child = ProofState("", "hard_child")
    fallback_child = ProofState("", "fallback_child")
    graph = ANDORGraph(root)
    stale = Action("skeleton", "stale", children=(hard_child,))
    fallback = Action("skeleton", "fallback", children=(fallback_child,))
    graph.expand(root, stale, r_env=1.0)
    graph.expand(root, fallback, r_env=1.0)
    graph.mark_failed(fallback)
    stats = {
        root: StateStats(
            depth=0,
            committed_skeleton=stale,
            reserved_skeletons=[(5.0, fallback)],
        )
    }
    queue = StatePriorityQueue()
    rollout = make_rollout(commit_stale_rounds_before_fallback=1)
    seen = {rollout.state_key(root): root}

    result = rollout.refresh_commitments(graph, stats, queue, seen)

    assert result["committed_stale"] == 1
    assert result["fallback_activated"] == 0
    assert graph.status(stale) == "OPEN"
    assert graph.status(fallback) == "FAILED"
    assert stats[root].committed_skeleton == stale
    assert len(queue) == 0


def test_stale_committed_skeleton_can_still_solve_without_parallel_fallback():
    root = ProofState("", "root")
    child = ProofState("", "child")
    fallback_child = ProofState("", "fallback_child")
    graph = ANDORGraph(root)
    stale = Action("skeleton", "stale", children=(child,))
    fallback = Action("skeleton", "fallback", children=(fallback_child,))
    graph.expand(root, stale, r_env=1.0)
    graph.expand(root, fallback, r_env=1.0)
    graph.mark_failed(fallback)
    stats = {
        root: StateStats(
            depth=0,
            committed_skeleton=stale,
            reserved_skeletons=[(5.0, fallback)],
        ),
        child: StateStats(depth=1),
    }
    queue = StatePriorityQueue()
    rollout = make_rollout(commit_stale_rounds_before_fallback=1)
    seen = {rollout.state_key(root): root, rollout.state_key(child): child}

    result = rollout.refresh_commitments(graph, stats, queue, seen)
    graph.mark_solved(child)
    rollout.propagate(graph, stats)

    assert result["fallback_activated"] == 0
    assert graph.status(stale) == "SOLVED"
    assert graph.status(fallback) == "FAILED"
    assert len(queue) == 0


def test_solved_committed_skeleton_does_not_activate_reserved_fallback():
    root = ProofState("", "root")
    child = ProofState("", "child")
    fallback_child = ProofState("", "fallback_child")
    graph = ANDORGraph(root)
    committed = Action("skeleton", "committed", children=(child,))
    fallback = Action("skeleton", "fallback", children=(fallback_child,))
    graph.expand(root, committed, r_env=1.0)
    graph.expand(root, fallback, r_env=1.0)
    graph.mark_solved(child)
    graph.mark_solved(committed)
    graph.mark_failed(fallback)
    stats = {
        root: StateStats(
            depth=0,
            committed_skeleton=committed,
            reserved_skeletons=[(5.0, fallback)],
        )
    }
    queue = StatePriorityQueue()
    rollout = make_rollout(commit_stale_rounds_before_fallback=1)
    seen = {rollout.state_key(root): root, rollout.state_key(child): child}

    result = rollout.refresh_commitments(graph, stats, queue, seen)

    assert result["fallback_activated"] == 0
    assert graph.status(root) == "SOLVED"
    assert graph.status(fallback) == "FAILED"
    assert len(queue) == 0


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
        root: StateStats(depth=0, committed_skeleton=skeleton),
        solved_child_1: StateStats(depth=1),
        solved_child_2: StateStats(depth=1),
        target: StateStats(depth=1, parent_skeletons=[(root, skeleton)]),
        unrelated: StateStats(depth=1),
    }
    scorer = SimpleHeuristicScorer()

    target_score = scorer.score_state(target, graph, stats)
    unrelated_score = scorer.score_state(unrelated, graph, stats)

    assert target_score > unrelated_score + 7.0


def test_uncommitted_skeleton_child_does_not_get_completion_bonus():
    root = ProofState("", "root")
    solved_child = ProofState("", "solved")
    target = ProofState("", "target")
    graph = ANDORGraph(root)
    skeleton = Action("skeleton", "split", children=(solved_child, target))
    graph.expand(root, skeleton, r_env=1.0)
    graph.add_state(solved_child, depth=1)
    graph.add_state(target, depth=1)
    graph.mark_solved(solved_child)
    stats = {
        root: StateStats(depth=0),
        solved_child: StateStats(depth=1),
        target: StateStats(depth=1, parent_skeletons=[(root, skeleton)]),
    }
    scorer = SimpleHeuristicScorer()

    assert scorer.committed_skeleton_progress_bonus(target, graph, stats) == 0.0


def test_finalize_unresolved_marks_open_states_and_actions_failed():
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
    assert graph.status(root) == "FAILED"
    assert graph.status(skeleton) == "FAILED"
    assert graph.status(child) == "FAILED"


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
        "skeleton_commitment",
    }
    assert meta["budget"]["used_total"] <= meta["budget"]["max_nodes"]
    assert meta["budget"]["used_tactic"] >= 2
    assert meta["budget"]["used_skeleton_raw"] == 1
    assert meta["skeleton_pipeline"]["requested"] == 1
    assert meta["skeleton_pipeline"]["inserted_raw"] == 1
    assert meta["skeleton_pipeline"]["selected_by_beam"] == 1
    assert meta["skeleton_pipeline"]["children_new"] == 1
    assert meta["skeleton_pipeline"]["children_duplicate"] == 1
    assert meta["skeleton_commitment"]["committed"] == 1
    assert meta["final_status"]["states"]["SOLVED"] >= 2
    assert meta["final_status"]["actions"]["skeleton_SOLVED"] == 1
    assert meta["depth_distribution"]["states_seen_by_depth"]["0"] == 1
