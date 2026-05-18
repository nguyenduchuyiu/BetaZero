import pytest

from gammazero.env.lean_env import LeanEnv
from gammazero.search.reward.calculator import RewardCalculator
from gammazero.search.reward.reward_assigner import DependencyRewardAssigner


# pytestmark = pytest.mark.integration


class _PatchedLean:
    def __init__(self, dep_analysis):
        self.dep_analysis = dep_analysis
        self.allowed_vars = None

    def analyze_dependencies(self, proof_code, allowed_vars=None, target_name=None):
        self.allowed_vars = set(allowed_vars or [])
        self.target_name = target_name
        return self.dep_analysis


class _ContextSensitiveLean:
    def __init__(self):
        self.allowed_vars_calls = []

    def analyze_dependencies(self, proof_code, allowed_vars=None, target_name=None):
        allowed = set(allowed_vars or [])
        self.allowed_vars_calls.append((allowed, target_name))
        analysis = {
            "core_solved": ["MAIN_GOAL", "h_target"],
            "core_failed": [],
            "benign": [],
            "malignant": [],
        }
        if "h_sibling" in allowed:
            analysis["benign"].append("h_sibling")
        if "h_scaffold" in allowed:
            analysis["malignant"].append("h_scaffold")
        return analysis


def _assigner():
    # `calculate_r_dep` only needs LeanEnv.analyze_dependencies,
    # so no scheduler is needed here.
    return DependencyRewardAssigner(LeanEnv(None), RewardCalculator())


@pytest.mark.parametrize(
    ("name", "full_code", "action_code", "expected"),
    [
        ("direct proof", "theorem my_theorem : True := by\n  trivial", "trivial", 1.0),
        (
            "unused local proof",
            "theorem my_theorem : True := by\n"
            "  have h_unused : True := by trivial\n"
            "  trivial",
            "have h_unused : True := by trivial\ntrivial",
            pytest.approx(2 / 3),
        ),
        (
            "used plus garbage locals",
            "theorem my_theorem : True := by\n"
            "  have h_used : True := by trivial\n"
            "  have h_unused : True := by trivial\n"
            "  have h_bad : True := by sorry\n"
            "  exact h_used",
            "have h_used : True := by trivial\n"
            "have h_unused : True := by trivial\n"
            "have h_bad : True := by sorry\n"
            "exact h_used",
            pytest.approx(2 / 4.5),
        ),
        ("main goal sorry", "theorem my_theorem : True := by\n  exact sorry", "exact sorry", 0.0),
    ],
)
def test_calculate_r_dep_scores_action_locals(name, full_code, action_code, expected):
    assigner = _assigner()

    score = assigner.calculate_r_dep(full_code, action_code)

    print(f"{name}: r_dep={score:.4f}")
    assert score == expected


def test_calculate_r_dep_can_target_action_snippet_only():
    assigner = _assigner()
    full_code = (
        "theorem my_theorem : True := by\n"
        "  have h_context_noise : True := by trivial\n"
        "  have h_used : True := by trivial\n"
        "  exact h_used"
    )
    action_code = "have h_used : True := by trivial\nexact h_used"

    score = assigner.calculate_r_dep(full_code, action_code)

    print(f"action snippet only: r_dep={score:.4f}")
    assert score == 1.0


def test_patched_subgoal_r_dep_must_score_marked_target_not_parent_scaffold():
    lean = _ContextSensitiveLean()
    assigner = DependencyRewardAssigner(lean, RewardCalculator())
    target_action = "have h_target : True := by sorry\nexact h_target"
    parent_scaffold_action = (
        "have h_scaffold : True := by trivial\n"
        "have h_target : True := by sorry\n"
        "have h_sibling : True := by sorry\n"
        "exact h_target"
    )
    target_full = (
        "theorem my_theorem : True := by\n"
        "  have h_target : True := by sorry\n"
        "  exact h_target"
    )
    parent_full = (
        "theorem my_theorem : True := by\n"
        "  have h_scaffold : True := by trivial\n"
        "  have h_target : True := by sorry\n"
        "  have h_sibling : True := by sorry\n"
        "  exact h_target"
    )

    target_score = assigner.calculate_patched_tactic_r_dep(target_full, target_action)
    polluted_score = assigner.calculate_patched_tactic_r_dep(
        parent_full,
        parent_scaffold_action,
    )
    targeted_parent_score = assigner.calculate_patched_tactic_r_dep(
        parent_full,
        target_action,
        target_name="h_target",
    )

    assert lean.allowed_vars_calls[0] == ({"h_target"}, None)
    assert lean.allowed_vars_calls[1] == ({"h_scaffold", "h_target", "h_sibling"}, None)
    assert lean.allowed_vars_calls[2] == ({"h_target"}, "h_target")
    assert target_score == 1.0
    assert polluted_score == pytest.approx(2 / 4.5)
    assert targeted_parent_score == 1.0


@pytest.mark.parametrize(
    ("name", "action_code", "dep_analysis", "expected"),
    [
        (
            "sorry local used by final proof",
            "have h : True := by sorry\nexact h",
            {
                "core_solved": ["MAIN_GOAL"],
                "core_failed": ["h"],
                "benign": [],
                "malignant": [],
            },
            1.0,
        ),
        (
            "sorry local not used",
            "have h : True := by sorry\ntrivial",
            {
                "core_solved": ["MAIN_GOAL"],
                "core_failed": [],
                "benign": [],
                "malignant": ["h"],
            },
            pytest.approx(1 / 3),
        ),
        (
            "naked exact sorry",
            "exact sorry",
            {
                "core_solved": [],
                "core_failed": ["MAIN_GOAL"],
                "benign": [],
                "malignant": [],
            },
            0.0,
        ),
    ],
)
def test_calculate_patched_tactic_r_dep_scores_sorry_scaffolding(
    name,
    action_code,
    dep_analysis,
    expected,
):
    lean = _PatchedLean(dep_analysis)
    assigner = DependencyRewardAssigner(lean, RewardCalculator())
    full_code = f"theorem my_theorem : True := by\n  {action_code.replace(chr(10), chr(10) + '  ')}"

    score = assigner.calculate_patched_tactic_r_dep(full_code, action_code)

    print(f"{name}: r_dep={score:.4f}, vars={sorted(lean.allowed_vars or [])}")
    assert score == expected
