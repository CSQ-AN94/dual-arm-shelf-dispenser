"""run() observation-leg orchestration (no hardware).

The guided teaching corridor was removed (2026-07-17): the head->observation
leg always goes through MoveIt free planning (_select_observation_flange +
_plan_flange). This exercises the plan-only orchestration with everything
below the state machine mocked out.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bottle_grasp.demo as demo_module
from bottle_grasp.core import Localization, SafetyAbort


def _make_demo(calls):
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)

    class Args:
        pass

    demo.args = Args()
    demo.args.execute = False
    demo.args.plan_only = True
    demo.args.resume_at_wrist = False
    demo.args.stop_after_observation = False
    demo.args.observe_seconds = 0
    demo.params = demo_module.DemoParams()

    class FakeCalibration:
        T_base_right_to_camera_head = np.eye(4)

    class FakeConfig:
        calibration = FakeCalibration()

    demo.cfg = FakeConfig()

    class FakeSafety:
        def assert_tcp_point(self, point, label=""):
            pass

    demo.safety = FakeSafety()
    demo.stage = lambda name, msg="": calls.append(("stage", name))
    demo.initialize = lambda: calls.append(("initialize",))
    demo._build_head_scene = lambda target: calls.append(("build_head_scene",))
    demo._select_observation_flange = lambda target: (
        calls.append(("select_flange",)),
        (np.eye(4), [0.0] * 7),
    )[1]
    demo._plan_flange = lambda name, flange, goal: (
        calls.append(("autonomous_plan", name)),
        {"points_deg": [[0.0] * 7]},
    )[1]
    demo._plan_observation = lambda target: (
        calls.append(("safe_observation_plan",)),
        {"points_deg": [[0.0] * 7]},
    )[1]
    loc = Localization(
        [0, 0, 0.5], [0, 0.6, -0.05], [320, 240], 0.5, 0.001, 0.002,
        [0, 0, 10, 10], 0.9, 7,
    )
    demo.localize = lambda *a, **k: (calls.append(("localize",)), loc)[1]
    return demo


def test_run_plans_observation_autonomously():
    calls = []
    demo = _make_demo(calls)
    demo.run()
    assert ("initialize",) in calls
    assert ("localize",) in calls
    assert ("build_head_scene",) in calls
    assert ("safe_observation_plan",) in calls


def test_fence_rejected_moveit_plan_is_replanned_before_execution():
    """Replay the real table-edge failure: reject plan 1, accept plan 2."""
    calls = []
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)
    demo.params = demo_module.DemoParams()
    demo.scene_voxels = []
    demo.scene_boxes = [{"id": "fence_table_top"}]
    demo.stage = lambda name, msg="": calls.append(("stage", name, msg))

    class FakeSafety:
        moveit_frame = "platform_base_link"

        @staticmethod
        def pose_to_moveit(pose):
            return pose

        @staticmethod
        def points_to_moveit(points):
            return list(points)

        @staticmethod
        def moveit_workspace():
            return {"min": [-1, -1, -1], "max": [1, 1, 1]}

    class FakeRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def controller_flange_from_joints(_joints):
            return np.eye(4)

        @staticmethod
        def validate_planned_joints(
            points, max_step, safety, start_joints_deg=None
        ):
            attempt = int(points[0][0])
            calls.append(("validate", attempt))
            if attempt == 1:
                raise SafetyAbort(
                    "轨迹 TCP 点 54 进入禁入区 table_top: "
                    "[0.2797, 0.3385, -0.4709]"
                )
            return 120

    class FakeLeftRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

    class FakePlanner:
        attempts = 0

        def plan(self, **kwargs):
            self.attempts += 1
            calls.append(("plan", self.attempts))
            return {
                "success": True,
                "points_deg": [[float(self.attempts)] * 7],
                "planning_time": 0.01,
            }

        @staticmethod
        def validate_exact_path(**kwargs):
            return {"checked_states": len(kwargs["points_deg"])}

    demo.safety = FakeSafety()
    demo.robot = FakeRobot()
    demo.left_robot = FakeLeftRobot()
    demo.planner = FakePlanner()

    plan = demo._plan_flange("moveit_observation", np.eye(4), [0.0] * 7)

    assert demo.planner.attempts == 2
    assert plan["points_deg"] == [[2.0] * 7]
    assert calls.index(("validate", 1)) < calls.index(("plan", 2))


def _precheck_demo(calls):
    """Demo stub with only what _observation_plan_targets touches."""
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)
    demo.params = demo_module.DemoParams()
    demo.stage = lambda name, msg="": calls.append(("stage", name, msg))

    class FakeCalibration:
        T_end_right_to_camera_rightwrist = np.eye(4)

    class FakeConfig:
        calibration = FakeCalibration()

    demo.cfg = FakeConfig()

    class FakeSafety:
        def assert_tcp_point(self, point, *, label=""):
            pass

    demo.safety = FakeSafety()
    return demo


def test_candidates_failing_even_the_hard_margin_are_rejected():
    """2026-07-18 真机复现：从端点出发的抓取接近段死于限位（任何余量档位
    都过不了）——预演必须在选观察位时就把它淘汰，档位要一路降到执行余量。"""
    calls = []
    demo = _precheck_demo(calls)
    margins_seen = []

    class NearLimitRobot:
        def joints_deg(self):
            return [0.0] * 7

        def solve_flange_ik(self, flange, params, seed_joints_deg=None):
            return [0.0, 129.2, 0.0, 30.0, 0.0, 0.0, 0.0]

        def plan_ik(self, poses, params, *, allow_first_jump=False,
                    seed_joints_deg=None):
            assert seed_joints_deg is not None
            margins_seen.append(params.joint_limit_margin_deg)
            raise SafetyAbort("路径点 1 关节 J2 距限位过近: 129.2°")

    demo.robot = NearLimitRobot()
    with pytest.raises(SafetyAbort, match="完整抓放预演全部失败"):
        demo._observation_plan_targets(np.array([0.0, 0.52, -0.11]))
    # 降级尝试必须覆盖从软余量到执行余量的全部档位，最宽的先试
    assert max(margins_seen) == demo.params.observation_grasp_margin_deg
    assert min(margins_seen) == demo.params.joint_limit_margin_deg


def test_observation_search_keeps_same_camera_pose_and_tries_j7_other_turn():
    """A bounded multi-turn J7 branch is not a 180deg camera-roll branch."""
    demo = _precheck_demo([])
    target = np.array([0.1712982892, 0.7017241153, -0.0875748415])
    flange = np.eye(4)
    demo._observation_flange_candidates = lambda _target: [flange]

    class MultiTurnRobot:
        @staticmethod
        def joints_deg():
            return [0.0, 0.0, 0.0, 30.0, 0.0, 0.0, -170.0]

        @staticmethod
        def solve_flange_ik_candidates(_flange, params, seed_joints_deg=None):
            return [
                [0.0, 0.0, 0.0, 30.0, 0.0, 0.0, -170.0],
                [0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 190.0],
            ]

    demo.robot = MultiTurnRobot()
    demo._grasp_precheck_margin = lambda *_args, **_kwargs: 10.0

    candidates = demo._observation_plan_targets(target)

    assert [candidate.goal_joints[6] for candidate in candidates] == [-170.0, 190.0]
    assert all(np.array_equal(candidate.flange, flange) for candidate in candidates)
    assert all("相机翻腕" not in candidate.label for candidate in candidates)


def test_observation_precheck_includes_moveit_whole_arm_collision_validation():
    """A controller-IK-valid local path can still put the hand through a shelf."""
    demo = _precheck_demo([])

    class IKOnlyRobot:
        @staticmethod
        def plan_ik(poses, params, *, allow_first_jump=False,
                    seed_joints_deg=None):
            return [[0.0] * 7 for _ in poses]

    demo.robot = IKOnlyRobot()
    validated = []

    def reject_hand_collision(**kwargs):
        validated.append(kwargs["name"])
        raise SafetyAbort("r_hand collides with fence_shelf_bottom")

    demo._validate_local_joint_path = reject_hand_collision
    target = SimpleNamespace(
        label="candidate",
        flange=np.eye(4),
        goal_joints=tuple([0.0] * 7),
    )

    assert not demo._grasp_precheck_ok(
        target, np.array([0.0, 0.52, -0.11]), 3.0
    )
    assert validated == ["observation_grasp_precheck"]


def test_shelf_precheck_does_not_retry_fixed_transit_rolls():
    """A rejected authored path is not retried at hard-coded roll angles."""
    demo = _precheck_demo([])
    planned_paths = []

    class IKRobot:
        @staticmethod
        def plan_ik(poses, params, *, allow_first_jump=False,
                    seed_joints_deg=None):
            planned_paths.append(list(poses))
            raise SafetyAbort("authored path collides with shelf lip")

    demo.robot = IKRobot()
    demo._validate_local_joint_path = lambda **_kwargs: None
    target = SimpleNamespace(
        label="candidate",
        flange=np.eye(4),
        goal_joints=tuple([0.0] * 7),
    )

    assert not demo._grasp_precheck_ok(
        target, np.array([0.0, 0.52, -0.11]), 3.0
    )
    assert len(planned_paths) == 1


def test_moveit_collision_is_not_replayed_at_three_joint_margin_levels():
    demo = _precheck_demo([])
    ik_calls = []
    validation_calls = []

    class IKRobot:
        @staticmethod
        def plan_ik(poses, params, *, allow_first_jump=False,
                    seed_joints_deg=None):
            ik_calls.append(params.joint_limit_margin_deg)
            return [[0.0] * 7 for _ in poses]

    demo.robot = IKRobot()

    def reject_collision(**_kwargs):
        validation_calls.append(1)
        raise SafetyAbort("r_hand collides with fence_shelf_bottom")

    demo._validate_local_joint_path = reject_collision
    target = SimpleNamespace(
        label="candidate",
        flange=np.eye(4),
        goal_joints=tuple([0.0] * 7),
    )

    assert demo._grasp_precheck_margin(
        target, np.array([0.0, 0.52, -0.11])
    ) is None
    assert ik_calls == [demo.params.observation_grasp_margin_deg]
    assert validation_calls == [1]


def test_plan_only_site_check_does_not_skip_local_moveit_validation():
    """The no-motion site check must exercise the same local collision seam.

    2026-07-20现场复现：site_check 在约 1 秒内把 16 个候选全部判为可行，
    随后的真实任务却花数分钟将相同候选判为 r_hand vs shelf_bottom。
    根因不能被 ``task_mode=None`` 静默短路。
    """
    demo = _precheck_demo([])
    demo.args = SimpleNamespace(task_mode=None, plan_only=True)
    demo.stop_event = __import__("threading").Event()
    demo.scene_voxels = []
    demo.scene_boxes = []

    class Robot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def validate_planned_joints(*_args, **_kwargs):
            return None

    class LeftRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

    class RejectingPlanner:
        @staticmethod
        def validate_exact_path(**_kwargs):
            raise SafetyAbort("r_hand collides with fence_shelf_bottom")

    class Safety:
        moveit_frame = "platform_base_link"

        @staticmethod
        def points_to_moveit(points):
            return list(points)

    demo.robot = Robot()
    demo.left_robot = LeftRobot()
    demo.planner = RejectingPlanner()
    demo.safety = Safety()

    with pytest.raises(SafetyAbort, match="r_hand.*shelf_bottom"):
        demo._validate_local_joint_path(
            name="observation_grasp_precheck",
            joints=[[1.0] * 7],
            target_base=np.array([0.0, 0.52, -0.11]),
            start_joints_deg=[0.0] * 7,
        )


def test_hypothetical_observation_precheck_starts_at_candidate_not_live_home():
    """Do not validate a fictitious live-home-to-grasp interpolation.

    The captured local_01 request began at the live home
    [7.665, 113.884, ...] even though controller IK had been seeded from the
    hypothetical observation candidate.  That invented path crossed the
    shelf and rejected every candidate before global transfer planning.
    """
    demo = _precheck_demo([])
    live_home = [7.0] * 7
    candidate = tuple([42.0] * 7)
    captured = []

    class Robot:
        @staticmethod
        def plan_ik(*_args, **_kwargs):
            return [[43.0] * 7]

    demo.robot = Robot()

    def validate(**kwargs):
        captured.append(tuple(kwargs["start_joints_deg"]))

    demo._validate_local_joint_path = validate
    target = SimpleNamespace(
        label="candidate",
        flange=np.eye(4),
        goal_joints=candidate,
    )

    assert demo._grasp_precheck_ok(
        target, np.array([0.0, 0.52, -0.11]), 3.0
    )
    assert captured == [candidate]
    assert captured[0] != tuple(live_home)


def test_stop_during_observation_precheck_aborts_immediately():
    """Ctrl+C is terminal inside the single precheck candidate."""
    import threading

    demo = _precheck_demo([])
    demo.stop_event = threading.Event()
    attempts = []

    class InterruptedRobot:
        def plan_ik(self, *_args, **_kwargs):
            attempts.append(1)
            demo.stop_event.set()
            raise SafetyAbort("MoveIt helper interrupted")

    demo.robot = InterruptedRobot()
    demo._validate_local_joint_path = lambda **_kwargs: None
    target = SimpleNamespace(
        label="candidate",
        flange=np.eye(4),
        goal_joints=tuple([0.0] * 7),
    )

    with pytest.raises(SafetyAbort, match="用户停止"):
        demo._grasp_precheck_ok(
            target, np.array([0.0, 0.52, -0.11]), 3.0
        )
    assert attempts == [1]


def test_invalid_observation_endpoint_is_checked_once_before_continuation():
    """An invalid observation state is rejected before grasp continuation.

    2026-07-21现场复现：候选 1/2 的 index 0 同时碰到 shelf_bottom 和
    slot_right_guard，但旧流程仍为每个端点运行 5 roll × 3 margin。端点姿态
        在这些尝试中完全不变，应先做一次状态复核并立即淘汰，且不得
        为了绕过碰撞而把相机光轴翻转 180°。
    """
    import threading

    demo = _precheck_demo([])
    demo.args = SimpleNamespace(task_mode=None, plan_only=True)
    demo.stop_event = threading.Event()
    demo.scene_voxels = []
    demo.scene_boxes = []
    demo._observation_flange_candidates = lambda _target: [np.eye(4)]
    validation_calls = []
    local_ik_calls = []

    class Robot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def solve_flange_ik(*_args, **_kwargs):
            return [10.0] * 7

        @staticmethod
        def validate_planned_joints(*_args, **_kwargs):
            return None

        @staticmethod
        def plan_ik(*_args, **_kwargs):
            local_ik_calls.append(1)
            return [[11.0] * 7]

    class LeftRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

    class Planner:
        @staticmethod
        def validate_exact_path(**kwargs):
            validation_calls.append(kwargs["name"])
            raise SafetyAbort("index 0: r_hand vs fence_shelf_bottom")

    class Safety:
        moveit_frame = "platform_base_link"

        @staticmethod
        def assert_tcp_point(*_args, **_kwargs):
            return None

        @staticmethod
        def points_to_moveit(points):
            return list(points)

    demo.robot = Robot()
    demo.left_robot = LeftRobot()
    demo.planner = Planner()
    demo.safety = Safety()

    with pytest.raises(SafetyAbort, match="所有右腕观察位候选"):
        demo._observation_plan_targets(np.array([0.0, 0.52, -0.11]))

    assert validation_calls == ["local_01_observation_endpoint_precheck"]
    assert local_ik_calls == []


def test_observation_candidates_keep_wrist_camera_pitch_moderate():
    demo = _precheck_demo([])
    target = np.array([0.0, 0.52, -0.11])

    candidates = demo._observation_flange_candidates(target)

    assert candidates
    pitches = [
        np.degrees(np.arcsin(np.clip(flange[2, 2], -1.0, 1.0)))
        for flange in candidates
    ]
    assert min(pitches) >= demo.params.observation_camera_min_pitch_deg
    assert max(pitches) <= demo.params.observation_camera_max_pitch_deg
    # Shelf runs need a raised observation family.  At 36--40 cm standoff it
    # looks down about 18--20 degrees: still moderate, but the old -15 degree
    # preference silently deleted it before IK/collision checks.
    cameras = [flange @ demo.T_flange_wrist_camera for flange in candidates]
    assert max(camera[2, 3] for camera in cameras) >= target[2] + 0.129
    first_camera = cameras[0]
    assert first_camera[2, 3] >= target[2] + 0.129


def test_observation_transfer_rejects_dive_below_endpoint_then_rise():
    demo = _precheck_demo([])

    class HeightRobot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def tcp_from_joints(joints):
            pose = np.eye(4)
            pose[2, 3] = float(joints[0])
            return pose

    demo.robot = HeightRobot()
    captured = {}

    def verified(_name, _targets, **kwargs):
        captured.update(kwargs)
        trajectory = {"points_deg": [[-0.20] * 7, [-0.10] * 7]}
        kwargs["trajectory_validator"](None, trajectory)
        raise AssertionError("expected vertical undershoot rejection")

    demo._verified_plan_targets = verified
    demo._observation_plan_targets = lambda _target, **_kwargs: []

    with pytest.raises(SafetyAbort, match="先下探再回升"):
        demo._plan_observation(np.array([0.0, 0.52, -0.11]))


def test_graspable_candidates_survive_and_keep_transfer_cost_order():
    calls = []
    demo = _precheck_demo(calls)

    class HealthyRobot:
        def __init__(self):
            self.solve_count = 0

        def joints_deg(self):
            return [0.0] * 7

        def solve_flange_ik(self, flange, params, seed_joints_deg=None):
            self.solve_count += 1
            return [float(self.solve_count % 5)] * 7

        def plan_ik(self, poses, params, *, allow_first_jump=False,
                    seed_joints_deg=None):
            return [[0.0] * 7 for _ in poses]

    demo.robot = HealthyRobot()
    targets = demo._observation_plan_targets(np.array([0.0, 0.52, -0.11]))
    assert targets
    assert all(target.goal_constraint == "joints" for target in targets)
    scores = [target.score for target in targets]
    assert scores == sorted(scores)
    assert any(
        "抓取预演可行" in msg for _, name, msg in calls if name == "生成右腕观察位候选"
    )


def test_observation_search_stops_after_enough_complete_task_candidates():
    demo = _precheck_demo([])
    demo.params = demo_module.DemoParams(
        observation_viable_candidate_limit=2
    )
    demo._observation_flange_candidates = lambda _target: [
        np.eye(4) for _ in range(20)
    ]
    endpoint_checks = []
    grasp_checks = []

    class Robot:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def solve_flange_ik(_flange, _params, seed_joints_deg=None):
            return [0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0]

    demo.robot = Robot()
    demo._validate_local_joint_path = (
        lambda **_kwargs: endpoint_checks.append(1)
    )

    def viable(*_args, **_kwargs):
        grasp_checks.append(1)
        return 10.0

    demo._grasp_precheck_margin = viable

    targets = demo._observation_plan_targets(np.array([0.0, 0.52, -0.11]))

    assert len(targets) == 2
    assert endpoint_checks == [1, 1]
    assert grasp_checks == [1, 1]


def test_staging_start_is_the_ik_seed_not_the_physical_hang_state():
    """The no-motion chained rehearsal must reproduce the post-stage branch.

    The 2026-07-19 failure was caused by seeding redundant observation IK
    from the natural hang (J2=121.6, J4=7.3).  Merely scoring candidates as if
    the arm were staged would still generate that same tucked branch.
    """
    demo = _precheck_demo([])
    demo._observation_flange_candidates = lambda _target: [np.eye(4)]
    demo._grasp_precheck_margin = lambda _target, _point: 10.0
    natural_hang = np.array(
        [-10.4, 121.6, 55.4, 7.3, 22.6, 5.8, -147.3]
    )
    staging = np.array(
        [25.21, 54.74, 21.496, 62.516, -40.634, -36.028, -23.643]
    )
    seeds = []

    class SeedAwareRobot:
        @staticmethod
        def joints_deg():
            return natural_hang.tolist()

        @staticmethod
        def solve_flange_ik(_flange, _params, seed_joints_deg=None):
            seeds.append(np.asarray(seed_joints_deg, dtype=float))
            return list(map(float, seed_joints_deg))

    demo.robot = SeedAwareRobot()

    targets = demo._observation_plan_targets(
        np.array([0.0, 0.52, -0.11]),
        current_joints_deg=staging,
    )

    assert len(targets) == 1
    np.testing.assert_allclose(seeds[0], staging)
    assert targets[0].score == pytest.approx(0.0)


def test_tight_margin_candidates_are_kept_but_ranked_after_roomy_ones():
    """2026-07-18 晚真机 watch 复现：10° 二元筛选把 11 个端点砍到 1 个，
    唯一幸存者 MoveIt 规划失败（error=99999）后没有任何备胎直接中止。
    分级录取必须保住窄余量候选作为后备，同时让宽余量的排前面。"""
    calls = []
    demo = _precheck_demo(calls)
    wide = demo.params.observation_grasp_margin_deg

    class MixedRobot:
        def __init__(self):
            self.solve_count = 0

        def joints_deg(self):
            return [0.0] * 7

        def solve_flange_ik(self, flange, params, seed_joints_deg=None):
            self.solve_count += 1
            # 第一个候选给最大的转移代价，其余递减——用来验证排序不只看代价
            return [float(50 - self.solve_count % 5)] * 7

        def plan_ik(self, poses, params, *, allow_first_jump=False,
                    seed_joints_deg=None):
            # 只有第一个解出的候选（转移代价最大）通过宽余量；
            # 其余候选只在执行余量（3°）下可行。
            first_candidate = seed_joints_deg[0] == 49.0
            if first_candidate:
                return [[0.0] * 7 for _ in poses]
            if params.joint_limit_margin_deg > demo.params.joint_limit_margin_deg:
                raise SafetyAbort("路径点 1 关节 J2 距限位过近")
            return [[0.0] * 7 for _ in poses]

    demo.robot = MixedRobot()
    targets = demo._observation_plan_targets(np.array([0.0, 0.52, -0.11]))
    # 窄余量候选全部保留（没有被一刀切淘汰）
    assert len(targets) > 1
    # 宽余量候选排第一；其后存在转移代价更小的窄余量候选——证明排序是
    # "余量档位优先于转移代价"，而不是单纯按代价排
    assert targets[0].goal_joints[0] == 49.0
    assert any(
        target.score < targets[0].score for target in targets[1:]
    )
