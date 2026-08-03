"""SafeMotionPlanner interface tests; no ROS or robot hardware."""

from pathlib import Path

import numpy as np
import pytest

from bottle_grasp.core import DemoParams, SafetyAbort, pose_matrix
from bottle_grasp.safe_planner import PlanTarget, SafeMotionPlanner
from bottle_grasp.safety import load_safety_profile


class FakeLeftRobot:
    @staticmethod
    def joints_deg():
        return [0.0] * 7


def _profile():
    return load_safety_profile(
        Path(__file__).parents[2] / "bottle_grasp" / "safety_profiles.json",
        "table_demo",
        require_verified=False,
    )


def _target(label: str, score: float, value: float) -> PlanTarget:
    return PlanTarget(
        label=label,
        flange=np.eye(4),
        goal_joints=tuple([value] * 7),
        score=score,
    )


def test_fence_violation_becomes_moveit_feedback_box():
    safety = _profile()

    class FakeMoveIt:
        def __init__(self):
            self.calls = []
            self.validations = 0
            self.validation_calls = []

        def plan(self, **kwargs):
            self.calls.append(kwargs)
            attempt = len(self.calls)
            return {"points_deg": [[float(attempt)] * 7]}

        def validate_exact_path(self, **kwargs):
            self.validations += 1
            self.validation_calls.append(kwargs)
            return {"checked_states": len(kwargs["points_deg"])}

    class FakeRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def controller_flange_from_joints(_joints):
            return np.eye(4)

        @staticmethod
        def validate_planned_joints(
            points, max_step, profile, start_joints_deg=None
        ):
            if int(points[0][0]) == 1:
                profile.assert_tcp_point(
                    [0.2797, 0.3385, -0.4709], label="轨迹 TCP 点 54"
                )
            return 120

    moveit = FakeMoveIt()
    planner = SafeMotionPlanner(
        moveit=moveit,
        robot=FakeRobot(),
        left_robot=FakeLeftRobot(),
        safety=safety,
        params=DemoParams(),
    )

    result = planner.plan(
        name="observe",
        targets=[_target("candidate", 0.0, 0.0)],
        obstacle_points=[],
        collision_boxes=safety.moveit_collision_boxes(),
    )

    assert result.attempts == 2
    assert result.checked_tcp_points == 120
    # Even the first SDK-fence-rejected trace must leave a MoveIt dense-state
    # artifact; the old early continue hid the decisive model evidence.
    assert moveit.validations == 2
    # The first postcheck must use the same scene that generated the path;
    # the SDK feedback box is only for the next planning request.
    assert moveit.validation_calls[0]["boxes"] == moveit.calls[0]["boxes"]
    assert moveit.validation_calls[1]["boxes"] == moveit.calls[1]["boxes"]
    assert len(moveit.calls[1]["boxes"]) == len(moveit.calls[0]["boxes"]) + 1
    assert moveit.calls[1]["boxes"][-1]["id"] == "replan_01"


def test_every_carrying_plan_and_dense_postcheck_include_held_guard():
    safety = _profile()
    guard = {
        "size": [0.09, 0.09, 0.27],
        "center": [0.0, 0.0, 0.2],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    class FakeMoveIt:
        def __init__(self):
            self.plan_calls = []
            self.validation_calls = []

        def plan(self, **kwargs):
            self.plan_calls.append(kwargs)
            return {"points_deg": [[1.0] * 7]}

        def validate_exact_path(self, **kwargs):
            self.validation_calls.append(kwargs)
            return {"checked_states": len(kwargs["points_deg"])}

    class FakeRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def controller_flange_from_joints(_joints):
            return np.eye(4)

        @staticmethod
        def validate_planned_joints(
            points, max_step, profile, start_joints_deg=None
        ):
            return len(points)

    moveit = FakeMoveIt()
    planner = SafeMotionPlanner(
        moveit=moveit,
        robot=FakeRobot(),
        left_robot=FakeLeftRobot(),
        safety=safety,
        params=DemoParams(),
        held_object=guard,
    )

    planner.plan(
        name="carry",
        targets=[_target("transport", 0.0, 1.0)],
        obstacle_points=[],
        collision_boxes=safety.moveit_collision_boxes(),
    )

    assert moveit.plan_calls
    assert moveit.validation_calls
    assert all(call["held_object"] is guard for call in moveit.plan_calls)
    assert all(
        call["held_object"] is guard for call in moveit.validation_calls
    )


def test_fence_feedback_is_shared_and_deduplicated_across_candidates():
    safety = _profile()

    class FakeMoveIt:
        def __init__(self):
            self.calls = []

        def plan(self, **kwargs):
            self.calls.append(kwargs)
            value = float(len(self.calls))
            return {"points_deg": [[value] * 7]}

        @staticmethod
        def validate_exact_path(**kwargs):
            return {"checked_states": len(kwargs["points_deg"])}

    class FakeRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def controller_flange_from_joints(_joints):
            return np.eye(4)

        @staticmethod
        def validate_planned_joints(
            points, _max_step, profile, start_joints_deg=None
        ):
            if points[0][0] in (1.0, 2.0):
                profile.assert_tcp_point(
                    [0.2797, 0.3385, -0.4709], label="same physical violation"
                )
            return 30

    moveit = FakeMoveIt()
    planner = SafeMotionPlanner(
        moveit=moveit,
        robot=FakeRobot(),
        left_robot=FakeLeftRobot(),
        safety=safety,
        params=DemoParams(
            global_plan_max_candidates=2,
            global_plan_attempts_per_candidate=2,
        ),
    )

    result = planner.plan(
        name="observe",
        targets=[_target("first", 0.0, 1.0), _target("second", 0.0, 2.0)],
        obstacle_points=[],
        collision_boxes=[],
    )

    assert result.attempts == 3
    assert [len(call["boxes"]) for call in moveit.calls] == [0, 1, 1]
    assert moveit.calls[1]["boxes"][0]["id"] == "replan_01"
    assert moveit.calls[2]["boxes"][0]["id"] == "replan_01"


def test_rejected_endpoint_falls_back_to_next_ranked_candidate():
    safety = _profile()

    class FakeMoveIt:
        def plan(self, **kwargs):
            value = float(kwargs["goal_joints_deg"][0])
            return {"points_deg": [[value] * 7]}

        @staticmethod
        def validate_exact_path(**kwargs):
            return {"checked_states": len(kwargs["points_deg"])}

    class FakeRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def controller_flange_from_joints(_joints):
            return np.eye(4)

        @staticmethod
        def validate_planned_joints(
            points, max_step, profile, start_joints_deg=None
        ):
            if int(points[0][0]) == 1:
                raise SafetyAbort("first endpoint path rejected")
            return 40

    planner = SafeMotionPlanner(
        moveit=FakeMoveIt(),
        robot=FakeRobot(),
        left_robot=FakeLeftRobot(),
        safety=safety,
        params=DemoParams(global_plan_attempts_per_candidate=1),
    )

    result = planner.plan(
        name="observe",
        targets=[_target("first", 1.0, 1.0), _target("second", 2.0, 2.0)],
        obstacle_points=[],
        collision_boxes=[],
    )

    assert result.target.label == "second"
    assert result.attempts == 2
    assert result.covered_candidates == 2
    assert result.total_candidates == 2
    assert result.planners_tried == ("RRTConnectkConfigDefault",)
    assert result.trajectory["search_coverage"] == {
        "attempted_candidates": 2,
        "total_candidates": 2,
        "planner_ids": ["RRTConnectkConfigDefault"],
        "attempts": 2,
    }


def test_candidate_priority_from_caller_is_preserved():
    """The caller's continuation-clearance order outranks transfer score."""
    safety = _profile()

    class FakeMoveIt:
        def plan(self, **kwargs):
            value = float(kwargs["goal_joints_deg"][0])
            return {"points_deg": [[value] * 7]}

        @staticmethod
        def validate_exact_path(**kwargs):
            return {"checked_states": len(kwargs["points_deg"])}

    class FakeRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def controller_flange_from_joints(_joints):
            return np.eye(4)

        @staticmethod
        def validate_planned_joints(*_args, **_kwargs):
            return 10

    planner = SafeMotionPlanner(
        moveit=FakeMoveIt(),
        robot=FakeRobot(),
        left_robot=FakeLeftRobot(),
        safety=safety,
        params=DemoParams(global_plan_attempts_per_candidate=1),
    )

    result = planner.plan(
        name="observe",
        targets=[
            _target("roomy-first", 100.0, 1.0),
            _target("cheap-second", -100.0, 2.0),
        ],
        obstacle_points=[],
        collision_boxes=[],
    )

    assert result.target.label == "roomy-first"


def test_plan_only_chain_can_use_an_explicit_hypothetical_start():
    safety = _profile()

    class FakeMoveIt:
        def __init__(self):
            self.start = None

        def plan(self, **kwargs):
            self.start = kwargs["start_joints_deg"]
            return {"points_deg": [[0.0] * 7]}

        @staticmethod
        def validate_exact_path(**kwargs):
            return {"checked_states": len(kwargs["points_deg"])}

    class FakeRobot:
        @staticmethod
        def joints_deg():
            return [99.0] * 7

        @staticmethod
        def controller_flange_from_joints(_joints):
            return np.eye(4)

        @staticmethod
        def validate_planned_joints(
            _points, _max_step, _profile, start_joints_deg=None
        ):
            assert start_joints_deg == pytest.approx([10.0] * 7)
            return 10

    moveit = FakeMoveIt()
    planner = SafeMotionPlanner(
        moveit=moveit,
        robot=FakeRobot(),
        left_robot=FakeLeftRobot(),
        safety=safety,
        params=DemoParams(global_plan_attempts_per_candidate=1),
    )

    result = planner.plan(
        name="observe_after_staging",
        targets=[_target("candidate", 0.0, 0.0)],
        obstacle_points=[],
        collision_boxes=[],
        start_right_joints_deg=[10.0] * 7,
    )

    assert moveit.start == pytest.approx([10.0] * 7)
    assert result.trajectory["start_joints_deg"] == pytest.approx(
        [10.0] * 7
    )


def test_replanning_is_bounded_and_reports_aggregate_rejections():
    safety = _profile()

    class FakeMoveIt:
        attempts = 0

        def plan(self, **kwargs):
            self.attempts += 1
            return {"points_deg": [[float(self.attempts)] * 7]}

        @staticmethod
        def validate_exact_path(**kwargs):
            return {"checked_states": len(kwargs["points_deg"])}

    class FakeRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def controller_flange_from_joints(_joints):
            return np.eye(4)

        @staticmethod
        def validate_planned_joints(
            points, max_step, profile, start_joints_deg=None
        ):
            raise SafetyAbort("still unsafe")

    moveit = FakeMoveIt()
    planner = SafeMotionPlanner(
        moveit=moveit,
        robot=FakeRobot(),
        left_robot=FakeLeftRobot(),
        safety=safety,
        params=DemoParams(
            global_plan_max_candidates=2,
            global_plan_attempts_per_candidate=2,
        ),
    )

    with pytest.raises(SafetyAbort, match="4 次安全规划") as failure:
        planner.plan(
            name="observe",
            targets=[_target("first", 1.0, 1.0), _target("second", 2.0, 2.0)],
            obstacle_points=[],
            collision_boxes=[],
        )

    assert moveit.attempts == 4
    assert "still unsafe" in str(failure.value)


def test_each_planner_pass_covers_candidates_before_retrying_one():
    safety = _profile()

    class NoRouteMoveIt:
        def __init__(self):
            self.searches = []

        def plan(self, **kwargs):
            self.searches.append(
                (
                    kwargs["planner_id"],
                    float(kwargs["goal_joints_deg"][0]),
                )
            )
            raise SafetyAbort("no route in this search slot")

    class StartOnlyRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

    moveit = NoRouteMoveIt()
    planner = SafeMotionPlanner(
        moveit=moveit,
        robot=StartOnlyRobot(),
        left_robot=FakeLeftRobot(),
        safety=safety,
        params=DemoParams(
            global_plan_max_candidates=2,
            global_plan_attempts_per_candidate=2,
            moveit_planner_ids=("planner-a", "planner-b"),
        ),
    )

    with pytest.raises(SafetyAbort, match="4 次安全规划"):
        planner.plan(
            name="observe",
            targets=[_target("first", 0.0, 1.0), _target("second", 0.0, 2.0)],
            obstacle_points=[],
            collision_boxes=[],
        )

    assert moveit.searches == [
        ("planner-a", 1.0),
        ("planner-a", 2.0),
        ("planner-b", 1.0),
        ("planner-b", 2.0),
    ]


def test_remaining_budget_is_shared_across_untried_candidates(monkeypatch):
    safety = _profile()
    clock = {"now": 100.0}
    monkeypatch.setattr(
        "bottle_grasp.safe_planner.time.monotonic",
        lambda: clock["now"],
    )

    class TimedNoRouteMoveIt:
        def __init__(self):
            self.searches = []

        def plan(self, **kwargs):
            allocated = float(kwargs["allowed_planning_time_s"])
            self.searches.append(
                (float(kwargs["goal_joints_deg"][0]), allocated)
            )
            clock["now"] += allocated
            raise SafetyAbort("time slice exhausted")

    class StartOnlyRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

    moveit = TimedNoRouteMoveIt()
    planner = SafeMotionPlanner(
        moveit=moveit,
        robot=StartOnlyRobot(),
        left_robot=FakeLeftRobot(),
        safety=safety,
        params=DemoParams(
            global_plan_max_candidates=3,
            global_plan_attempts_per_candidate=1,
            global_plan_search_budget_s=12.0,
            moveit_allowed_planning_time_s=6.0,
            moveit_planner_ids=("planner-a",),
        ),
    )

    with pytest.raises(SafetyAbort):
        planner.plan(
            name="observe",
            targets=[
                _target("first", 0.0, 1.0),
                _target("second", 0.0, 2.0),
                _target("third", 0.0, 3.0),
            ],
            obstacle_points=[],
            collision_boxes=[],
        )

    assert [value for value, _ in moveit.searches] == [1.0, 2.0, 3.0]
    assert [slot for _, slot in moveit.searches] == pytest.approx(
        [4.0, 4.0, 4.0]
    )


def test_moveit_post_validation_rejection_also_replans():
    safety = _profile()

    class FakeMoveIt:
        attempts = 0
        validations = 0

        def plan(self, **kwargs):
            self.attempts += 1
            return {"points_deg": [[float(self.attempts)] * 7]}

        def validate_exact_path(self, **kwargs):
            self.validations += 1
            if self.validations == 1:
                raise SafetyAbort("full-arm collision at state 3")
            return {"checked_states": len(kwargs["points_deg"])}

    class FakeRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def controller_flange_from_joints(_joints):
            return np.eye(4)

        @staticmethod
        def validate_planned_joints(
            points, max_step, profile, start_joints_deg=None
        ):
            return 10

    moveit = FakeMoveIt()
    planner = SafeMotionPlanner(
        moveit=moveit,
        robot=FakeRobot(),
        left_robot=FakeLeftRobot(),
        safety=safety,
        params=DemoParams(),
    )

    result = planner.plan(
        name="observe",
        targets=[_target("candidate", 0.0, 0.0)],
        obstacle_points=[],
        collision_boxes=[],
    )

    assert result.attempts == 2
    assert moveit.validations == 2


def test_duplicate_route_does_not_skip_later_planner_for_candidate():
    safety = _profile()

    class FakeMoveIt:
        def __init__(self):
            self.planners = []

        def plan(self, **kwargs):
            self.planners.append(kwargs["planner_id"])
            value = 1.0 if len(self.planners) < 3 else 3.0
            return {"points_deg": [[value] * 7]}

        @staticmethod
        def validate_exact_path(**kwargs):
            return {"checked_states": len(kwargs["points_deg"])}

    class FakeRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def controller_flange_from_joints(_joints):
            return np.eye(4)

        @staticmethod
        def validate_planned_joints(
            points, _max_step, _profile, start_joints_deg=None
        ):
            if points[0][0] == 1.0:
                raise SafetyAbort("first route is unsafe")
            return 20

    moveit = FakeMoveIt()
    planner = SafeMotionPlanner(
        moveit=moveit,
        robot=FakeRobot(),
        left_robot=FakeLeftRobot(),
        safety=safety,
        params=DemoParams(
            global_plan_attempts_per_candidate=3,
            moveit_planner_ids=("planner-a", "planner-b", "planner-c"),
        ),
    )

    result = planner.plan(
        name="observe",
        targets=[_target("candidate", 0.0, 0.0)],
        obstacle_points=[],
        collision_boxes=[],
    )

    assert result.attempts == 3
    assert moveit.planners == ["planner-a", "planner-b", "planner-c"]


def test_failed_continuation_eliminates_endpoint_but_keeps_other_candidates():
    safety = _profile()

    class FakeMoveIt:
        def __init__(self):
            self.values = []

        def plan(self, **kwargs):
            value = float(kwargs["goal_joints_deg"][0])
            self.values.append(value)
            if value == 2.0:
                raise SafetyAbort("second candidate has no global route")
            return {"points_deg": [[value] * 7]}

        @staticmethod
        def validate_exact_path(**_kwargs):
            raise AssertionError("continuation rejection precedes dense checks")

    class FakeRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def controller_flange_from_joints(_joints):
            return np.eye(4)

        @staticmethod
        def validate_planned_joints(*_args, **_kwargs):
            raise AssertionError("continuation rejection precedes fence checks")

    continuation_checks = []

    def continuation_validator(target, trajectory):
        continuation_checks.append(
            (target.label, float(trajectory["points_deg"][-1][0]))
        )
        raise SafetyAbort("observe endpoint enters J4 singularity on approach")

    moveit = FakeMoveIt()
    planner = SafeMotionPlanner(
        moveit=moveit,
        robot=FakeRobot(),
        left_robot=FakeLeftRobot(),
        safety=safety,
        params=DemoParams(
            global_plan_max_candidates=2,
            global_plan_attempts_per_candidate=2,
            moveit_planner_ids=("planner-a", "planner-b"),
        ),
    )

    with pytest.raises(SafetyAbort, match="J4 singularity"):
        planner.plan(
            name="observe",
            targets=[_target("singular", 0.0, 1.0), _target("blocked", 0.0, 2.0)],
            obstacle_points=[],
            collision_boxes=[],
            continuation_validator=continuation_validator,
        )

    assert continuation_checks == [("singular", 1.0)]
    assert moveit.values == [1.0, 2.0, 2.0]


def test_moveit_endpoint_must_mean_same_pose_to_controller_model():
    safety = _profile()

    class FakeMoveIt:
        attempts = 0

        def plan(self, **_kwargs):
            self.attempts += 1
            return {"points_deg": [[float(self.attempts)] * 7]}

        @staticmethod
        def validate_exact_path(**_kwargs):
            raise AssertionError("model mismatch must reject before path checks")

    class FakeRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def controller_flange_from_joints(_joints):
            flange = np.eye(4)
            flange[0, 3] = 0.05
            return flange

        @staticmethod
        def validate_planned_joints(*_args, **_kwargs):
            raise AssertionError("model mismatch must reject before fence checks")

    moveit = FakeMoveIt()
    planner = SafeMotionPlanner(
        moveit=moveit,
        robot=FakeRobot(),
        left_robot=FakeLeftRobot(),
        safety=safety,
        params=DemoParams(global_plan_attempts_per_candidate=2),
    )

    with pytest.raises(SafetyAbort, match="端点运动学不一致"):
        planner.plan(
            name="observe",
            targets=[_target("candidate", 0.0, 0.0)],
            obstacle_points=[],
            collision_boxes=[],
        )

    # A model disagreement is candidate-level, not something random route
    # retries can repair.
    assert moveit.attempts == 1


def test_runtime_same_state_fk_contract_rejects_frame_mismatch():
    class ContractMoveIt:
        enforces_model_contract = True

    class IdentitySafety:
        @staticmethod
        def pose_to_moveit(pose):
            return np.asarray(pose, dtype=float)

    class IdentityRobot:
        @staticmethod
        def controller_flange_from_joints(_joints):
            return np.eye(4)

    planner = SafeMotionPlanner(
        moveit=ContractMoveIt(),
        robot=IdentityRobot(),
        left_robot=FakeLeftRobot(),
        safety=IdentitySafety(),
        params=DemoParams(),
    )
    expected_link7 = [
        0.0,
        0.0,
        -planner.params.moveit_link7_to_controller_flange_m,
    ]
    trajectory = {
        "points_deg": [[0.0] * 7],
        "start_link7_fk": {
            "position": expected_link7,
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "endpoint_link7_fk": {
            "position": [0.05, *expected_link7[1:]],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }

    with pytest.raises(SafetyAbort, match="同状态 FK 不一致"):
        planner._assert_runtime_fk_contract([0.0] * 7, trajectory)


def test_g2_same_state_fk_replay_accepts_start_and_home_without_relaxing_guard():
    """Replay G2 run 20260724_191151_294662_490a95ed.

    The literals are the recorded controller/MoveIt poses at the exact same
    start and home joint states.  They catch a stale base-frame bridge without
    changing the 25 mm runtime model-disagreement ceiling.
    """

    safety = load_safety_profile(
        Path(__file__).parents[2] / "bottle_grasp" / "safety_profiles.json",
        "shelf_template",
        require_verified=True,
    )
    start_joints = [
        24.469999313354492,
        117.9280014038086,
        -48.21900177001953,
        43.801998138427734,
        20.719999313354492,
        -18.613000869750977,
        -61.233001708984375,
    ]
    home_joints = list(safety.home_joints_deg)
    endpoint_joints = [
        7.820853900536894,
        113.98483114074917,
        -7.829631514661014,
        34.129688943549986,
        -82.06358348317445,
        -83.99390505943447,
        -13.195059325359763,
    ]
    link7_to_flange = np.eye(4)
    link7_to_flange[2, 3] = 0.0172
    flange_to_tcp = np.eye(4)
    flange_to_tcp[2, 3] = 0.151

    # Recorded active controller TCP at the no-motion abort.
    start_tcp = pose_matrix(
        [0.251325, 0.006088, -0.567973, -3.028, -0.112, -1.332]
    )
    start_flange = start_tcp @ np.linalg.inv(flange_to_tcp)
    goal_flange = np.array(
        [
            [
                0.022734478256933394,
                -0.9933546873455141,
                -0.1128255672579615,
                0.1977437287569046,
            ],
            [
                -0.08584472389791004,
                -0.11437758828644767,
                0.9897214005348512,
                0.1896650493144989,
            ],
            [
                -0.9960491086674776,
                -0.012815319990996499,
                -0.08787457366184914,
                -0.2865840792655945,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    trajectory = {
        "points_deg": [endpoint_joints],
        "start_link7_fk": {
            "position": [
                -0.287415494310836,
                -0.13201681261202497,
                -0.38138059172959216,
            ],
            "quaternion_xyzw": [
                0.6137149856690378,
                0.7855024553012228,
                -0.009479888011292013,
                -0.07905656711682264,
            ],
        },
        "endpoint_link7_fk": {
            "position": [
                -0.2640182577966927,
                -0.2569207150955354,
                -0.28864221828661984,
            ],
            "quaternion_xyzw": [
                0.4882250100646429,
                0.5519311156609414,
                -0.4531272281417993,
                0.5016812715551352,
            ],
        },
    }

    class ReplayMoveIt:
        enforces_model_contract = True

        @staticmethod
        def plan(**_kwargs):
            return trajectory

        @staticmethod
        def validate_exact_path(**kwargs):
            return {"checked_states": len(kwargs["points_deg"])}

    class ReplayRobot:
        @staticmethod
        def joints_deg():
            return start_joints

        @staticmethod
        def controller_flange_from_joints(joints):
            if np.allclose(joints, start_joints, atol=1e-9, rtol=0.0):
                return start_flange
            return goal_flange

        @staticmethod
        def validate_planned_joints(
            points, _max_step, _profile, start_joints_deg=None
        ):
            assert start_joints_deg == start_joints
            return len(points)

    params = DemoParams(
        global_plan_attempts_per_candidate=1,
        global_plan_max_candidates=1,
    )
    assert params.moveit_endpoint_position_tolerance_m == 0.025
    planner = SafeMotionPlanner(
        moveit=ReplayMoveIt(),
        robot=ReplayRobot(),
        left_robot=FakeLeftRobot(),
        safety=safety,
        params=params,
        link7_to_controller_flange=link7_to_flange,
    )

    verified = planner.plan(
        name="moveit_return_home",
        targets=[
            PlanTarget(
                label="固定目标",
                flange=goal_flange,
                goal_joints=tuple(home_joints),
                goal_constraint="joints",
            )
        ],
        obstacle_points=[],
        collision_boxes=[],
    )

    assert verified.attempts == 1


def test_runtime_fk_contract_rejects_nonfinite_sdk_pose():
    class ContractMoveIt:
        enforces_model_contract = True

    class IdentitySafety:
        @staticmethod
        def pose_to_moveit(pose):
            return np.asarray(pose, dtype=float)

    class BadRobot:
        @staticmethod
        def controller_flange_from_joints(_joints):
            result = np.eye(4)
            result[0, 3] = np.nan
            return result

    planner = SafeMotionPlanner(
        moveit=ContractMoveIt(),
        robot=BadRobot(),
        left_robot=FakeLeftRobot(),
        safety=IdentitySafety(),
        params=DemoParams(),
    )
    fk = {
        "position": [0.0, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    with pytest.raises(SafetyAbort, match="SDK 起点 FK"):
        planner._assert_runtime_fk_contract(
            [0.0] * 7,
            {"points_deg": [[0.0] * 7], "start_link7_fk": fk, "endpoint_link7_fk": fk},
        )
