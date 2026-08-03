"""Shared data types and geometry helpers for the bottle grasp demo."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


class SafetyAbort(RuntimeError):
    """A fail-closed condition; no automatic retreat is allowed."""


class CameraFrameUnavailable(SafetyAbort):
    """The RGB-D stream is stale or missing; motion must stop."""


class BottleDetectionLost(SafetyAbort):
    """A live frame no longer contains a detector-approved bottle."""


class InsufficientDepth(SafetyAbort):
    """A fresh depth frame exists but cannot support a physical measurement."""


def stop_reason(stop_event) -> str:
    """Describe why a shared stop_event fired, distinguishing operator stop
    from an automatic guard trip.

    Guards that stop the task from a background thread (LeftArmStabilityGuard,
    _execute_plan's left-arm snapshot monitor) tag the event with a `.source`
    attribute before calling `.set()`.  Every caller that merely polls
    `stop_event.is_set()` used to raise an indistinguishable generic
    "用户停止", which reads as an operator Ctrl+C/STOP even when the real
    cause was an automatic safety guard tripping mid-task (2026-07-19: this
    made a left-arm-drift trip during post-place gripper close look like an
    unexplained hardware failure).
    """
    source = getattr(stop_event, "source", None)
    return "用户停止" if not source else f"自动安全停止（{source}）"


@dataclass
class DemoParams:
    samples: int = 7
    confidence: float = 0.45
    min_depth_m: float = 0.12
    max_depth_m: float = 0.65
    max_depth_mad_m: float = 0.018
    max_position_spread_m: float = 0.025
    max_relocalization_jump_m: float = 0.035
    # 续抓模式首次定位对保存先验的容差：桌子/瓶子可能被人为挪动过，
    # 放宽到12cm；接近过程中的分段跳变检查仍用 max_relocalization_jump_m。
    resume_prior_jump_m: float = 0.20
    tcp_z_m: float = 0.151
    moveit_link7_to_controller_flange_m: float = 0.0172
    tool_guard_xy_m: float = 0.10
    tool_guard_length_m: float = 0.19
    tool_guard_center_z_m: float = 0.095
    # RGB-D returns a point on the visible bottle surface, while the TCP is a
    # gripper reference point.  Stopping exactly at the visual point inserted
    # the fingers/palm too deeply in the 2026-07-19 supervised run.  Keep the
    # final TCP this far back along the approach axis.
    # 2026-07-19 supervised adjustment: 4 cm was slightly too shallow; move
    # 1 cm farther toward the bottle while retaining 3 cm against over-insertion.
    grasp_stop_short_m: float = 0.030
    pregrasp_standoff_m: float = 0.085
    # The observation wrist pose can be collision-free while a direct
    # observation->pregrasp interpolation still sweeps the rotating hand down
    # through a shelf lip.  Preserve the known-safe observation orientation,
    # rise vertically first, then turn and translate toward the bottle.
    # 2026-07-21 zero-motion replay: both surviving shelf candidates failed
    # only during that early rotational sweep against shelf_bottom.
    local_clearance_lift_m: float = 0.050
    # Post-release clearance is a different requirement from the pregrasp
    # hover.  It used to reuse 8.5 cm and left the open fingers too close to
    # the bottle; retreat farther after release without moving the pregrasp.
    retreat_standoff_m: float = 0.150
    segment_m: float = 0.045
    lift_m: float = 0.05
    # Three motion regimes with genuinely different risk, so they get
    # separate knobs instead of one shared number:
    #   transit_speed — the global MoveIt leg to the observation pose.  Fully
    #     collision-validated, in free space, nowhere near the bottle.  This
    #     is the leg that dominates cycle time (2026-07-18: 146 blocking
    #     movej points took ~90 s at 3%).
    #   travel_speed  — local straight-line approach toward the pregrasp
    #     hover point and the post-release retreat.  Both stay away from
    #     contact and stays at 15% even when global transit is faster.
    #   final_speed   — final approach, lift and lower; contact-adjacent.
    # 2026-07-22, operator-approved: global transit raised 75% -> 100% after
    # the complete MoveIt trajectory, dense independent fence checks, and the
    # SDK connect=1 blended execution path were validated on the shelf.  The
    # local travel and contact-adjacent approach/lift/lower legs intentionally
    # remain at 15%.
    transit_speed: int = 100
    travel_speed: int = 15
    final_speed: int = 15
    j4_singularity_deg: float = 8.0
    # 起点已在 J4≈0 奇异带内时，先用关节空间 movej 把肘弯到这个角度再做
    # 笛卡尔规划。比奇异带宽出 6°，避免逃逸后又贴着带边被下一次检查拒绝。
    j4_escape_deg: float = 14.0
    joint_limit_margin_deg: float = 3.0
    corridor_radius_m: float = 0.045
    obstacle_min_points: int = 18
    # The target itself is occupied space, not a generic clearance hole.  The
    # grasp point is roughly two-thirds down the bottle; these asymmetric
    # bounds cover the physical cylinder while leaving the measured table and
    # neighbouring objects outside it visible to the local corridor gate.
    target_occupancy_radius_m: float = 0.038
    target_occupancy_above_grasp_m: float = 0.18
    target_occupancy_below_grasp_m: float = 0.055
    target_occupancy_box_pad_px: int = 8
    # Transparent/curved bottle pixels span the front and rear surfaces.  The
    # locked robust depth may sit near one side of that span, so the wrist-box
    # target mask uses one physical radius plus this small sensor slack along
    # the optical axis.  It remains bounded enough to retain the 7.5 cm
    # foreground-obstacle regression.
    target_occupancy_depth_slack_m: float = 0.015
    frame_timeout_s: float = 1.0
    head_min_depth_m: float = 0.25
    head_max_depth_m: float = 2.2
    scene_voxel_m: float = 0.065
    scene_max_voxels: int = 550
    # 目标定位要 7 帧共识才敢信，环境点云却只用一帧——同一次运行里两套
    # 可靠性标准不一致，而"障碍物在哪/桌子多高"出错的后果同样是真实的。
    # 障碍体素取多帧并集（任一帧看到即占据，绝不投票删除闪烁的障碍物），
    # 桌面高度取多帧中位数且要求帧间一致，不一致说明采集期间场景在动。
    scene_samples: int = 3
    table_fit_agreement_m: float = 0.015
    # Kept for config compatibility only.  Global scene construction must not
    # erase a clearance sphere around the target: the 2026-07-18 value (14 cm)
    # could delete real obstacles beside the bottle.  Contact semantics belong
    # to the later local grasp stage, not the observation-transfer scene.
    scene_target_clearance_m: float = 0.0
    scene_image_bottom_crop: int = 405
    planned_joint_step_deg: float = 1.5
    # A MoveIt path is executed by the SDK adapter during the migration to a
    # FollowJointTrajectory controller.  Refuse stale starts, endpoint model
    # disagreement and blocking movej feedback that misses its command.
    planned_start_tolerance_deg: float = 0.8
    planned_tracking_tolerance_deg: float = 1.2
    # A collision scene is a planning snapshot, not a timeless fact.  A plan
    # that took too long to produce is refused before any global motion.
    scene_max_age_s: float = 900.0
    # 2026-07-19 实测：SDK rm_algo 与 URDF 在 5 组关节角上的同状态 FK 位置差
    # 随构型变化 1.8~17.4mm（旋转一致到 0.01°），是两套运动学模型的固有几何
    # 差异。容差必须盖过 17.4mm 才不会误杀；25mm 仍足以抓住坐标系级错误
    # （修复前的镜像变换差了 530mm/180°）。
    moveit_endpoint_position_tolerance_m: float = 0.025
    moveit_endpoint_orientation_tolerance_deg: float = 4.0
    # 每轮从头部点云拟合真实桌面并在容差内自适应 table_top 围栏。
    # 容差是"底盘每轮停靠位置的正常波动"量级；超出说明物理布置真的变了，
    # 必须重新走 runbook 测量流程而不是让软件猜。
    table_fit_min_below_m: float = 0.03
    table_fit_max_below_m: float = 0.40
    table_fit_horizontal_radius_m: float = 0.70
    table_fit_min_inliers: int = 60
    table_fit_height_tolerance_m: float = 0.12
    table_fit_edge_margin_m: float = 0.08
    table_fit_zone_attach_band_m: float = 0.06
    # 每轮从头部点云拟合货架各面（bottom/top/back/left/right），泛化自上面的
    # 桌面拟合同一套"分箱众数+多帧一致性+只扩不缩"算法（bottle_grasp/
    # shelf_model.py）。跟桌面不同：货架从未在真机上测量/验证过，这些默认值
    # 是工程判断的起点，不是实测反推出来的，第一次现场测量后应该重新核实。
    shelf_fit_min_gap_m: float = 0.02
    shelf_fit_max_gap_m: float = 0.35
    shelf_fit_lateral_radius_m: float = 0.35
    shelf_fit_min_inliers: int = 40
    shelf_fit_bound_tolerance_m: float = 0.05
    shelf_fit_edge_margin_m: float = 0.03
    shelf_fit_conservative_margin_m: float = 0.01
    # 视觉选目标：给定时只在这些 YOLO 类别名里选（按商品识别选格位）；不给
    # 则保持现状——detector.aliases 里的通用瓶子类别，桌面 demo 行为不变。
    target_product_classes: tuple[str, ...] | None = None
    # Global MoveIt transfer: try several endpoint candidates and feed an
    # independently detected fence violation back as a temporary collision
    # box. The bounds keep failure deterministic instead of retrying forever.
    # Exhaust the complete 4x4x3 observation lattice (48 poses) subject to a
    # wall-clock budget.  Each endpoint is tried with several genuinely
    # different OMPL planners; "two random calls failed" is not evidence that
    # no route exists.
    global_plan_max_candidates: int = 48
    # Stop the expensive endpoint + full-pick rehearsal once this many
    # genuinely executable observation alternatives have been found.  More
    # IK branches are useful redundancy, but validating every equivalent
    # branch before trying any complete task made the zero-motion check take
    # minutes after RM75 arm-angle search was enabled.
    observation_viable_candidate_limit: int = 4
    global_plan_attempts_per_candidate: int = 4
    global_plan_search_budget_s: float = 120.0
    moveit_allowed_planning_time_s: float = 6.0
    moveit_num_planning_attempts: int = 12
    moveit_planner_ids: tuple[str, ...] = (
        "RRTConnectkConfigDefault",
        "LBKPIECEkConfigDefault",
        "RRTstarkConfigDefault",
        "PRMstarkConfigDefault",
    )
    # 选观察位时预演后续抓取接近段用的软限位余量。2026-07-18 真机 observe：
    # 端点只按 3° 硬余量过关，选出 J2=129.2°（距限位 3.3°）的观察位，到位
    # 后 5 个抓取 roll 全部死于"J2 距限位过近"。预检余量必须显著大于硬
    # 余量，给"头部定位→腕部精定位"之间约 3cm 的目标漂移留出关节空间。
    observation_grasp_margin_deg: float = 10.0
    # Keep the wrist camera visually near level at the observation endpoint.
    # Negative optical-axis pitch looks down in the controller base frame.
    # A +13 cm shelf-clearance view at 36--40 cm standoff needs roughly
    # 18--20 degrees of downward pitch.  The previous -15 degree preference
    # discarded every raised endpoint before IK or collision checking and
    # left only hand-vs-shelf-bottom poses for the 2026-07-21 right-side item.
    observation_camera_min_pitch_deg: float = -22.0
    observation_camera_max_pitch_deg: float = 10.0
    # A transfer may descend when the observation endpoint is physically
    # lower, but it must not dive below both endpoints and then come back up.
    observation_vertical_undershoot_tolerance_m: float = 0.008
    replan_exclusion_size_m: float = 0.10
    head_width: int = 848
    head_height: int = 480
    wrist_relocalization_samples: int = 3
    # The current SDK execution path is blocking rm_movej/movel, not a streaming
    # controller.  The first closed-loop improvement is therefore deliberately
    # low-rate: reobserve at the verified pregrasp hover, make one bounded
    # translation, then reobserve.  These limits are small compared with the
    # 8.5 cm hover and prevent a noisy eye-in-hand estimate from turning into a
    # large unplanned chase.  Orientation correction is intentionally disabled.
    visual_servo_max_corrections: int = 2
    visual_servo_max_step_m: float = 0.008
    visual_servo_max_total_m: float = 0.015
    visual_servo_convergence_m: float = 0.004
    visual_servo_divergence_tolerance_m: float = 0.002
    visual_servo_max_position_spread_m: float = 0.012
    # Lift/release confirmation is already conditioned on a completed robot
    # action and then checked again for the expected motion direction.  With
    # three fresh fixed-head frames, require a strict 2/3 majority so one
    # gripper-occluded depth outlier cannot veto an otherwise proven action.
    # This does not relax initial target localization, which keeps the generic
    # 70%-and-at-least-3 rule below.
    post_action_confirmation_min_frames: int = 2
    # When wrist detection genuinely disappears during the observation-to-
    # pregrasp transit, the fixed head camera independently confirms that the
    # bottle is still close to the locked base-frame point.  It never silently
    # rewrites the target; a larger shift stops the motion.
    head_confirmation_tolerance_m: float = 0.05
    # Gripper feedback alone cannot prove the bottle moved.  Independent fixed
    # head RGB-D must observe a meaningful departure from the table lock and a
    # target meaningfully above the table lock before the task may become HELD.
    # The gripper hides the lower bottle after grasping, so the box-relative
    # depth sample can move upward on the bottle itself; compare direction and
    # horizontal association, not exact distance to locked+5cm.
    lift_confirmation_min_displacement_m: float = 0.025
    lift_confirmation_max_horizontal_m: float = 0.050
    # A detector-free post-lift pass is allowed only when three distinct wrist
    # frames show a nearer occluding surface at the expected target projection.
    # The fallback then re-reads three independent gripper states and verifies
    # the measured TCP motion instead of treating camera loss as occlusion.
    lift_occlusion_required_frames: int = 3
    lift_occlusion_depth_margin_m: float = 0.020
    lift_occlusion_roi_min_valid_depth: int = 18
    lift_occlusion_roi_min_valid_fraction: float = 0.80
    lift_occlusion_roi_min_foreground_fraction: float = 0.80
    lift_gripper_stability_position_tolerance: int = 6
    lift_tcp_vertical_tolerance_m: float = 0.015
    lift_tcp_max_horizontal_m: float = 0.015
    # Generic provisional bottle envelope used before lift.  Delivery
    # profiles may override these with their independently measured values.
    held_bottle_height_m: float = 0.25
    held_bottle_diameter_m: float = 0.07
    held_bottle_guard_padding_m: float = 0.02
    # Minimum fallback only.  The provisional guard derives the conservative
    # bottle bottom from height × (1 - grasp_height_fraction).
    bottle_bottom_below_tcp_m: float = 0.04
    # Release likewise requires a fresh 3-D measurement at the original lock;
    # it may never be synthesized from the lock's prior depth.  Wrist and head
    # detections can sample different heights on the bottle, so association is
    # horizontal while a separate lifted->released drop proves it came down.
    release_confirmation_tolerance_m: float = 0.035
    release_confirmation_min_drop_m: float = 0.025
    # 抓取点高度：检测框顶部向下的比例（0=瓶盖, 1=瓶底）。固定比例
    # 避免深度像素分布让高度逐帧漂移。0.45 是桌面/旧入口默认值；货架
    # profile 可按手掌包络和层板净空单独覆盖，不能让桌面参数绑架货架。
    grasp_height_fraction: float = 0.45
    # RM Plus two-finger gripper.  The legacy rm_set_gripper_* API does not
    # control the installed ZX gripper.
    gripper_open_position: int = 900
    gripper_close_position: int = 0
    gripper_speed: int = 100
    gripper_force: int = 30
    # 空夹动态标定只做与本轮实测张开位的粗行程检查，不再拿历史闭合位置
    # 当裁判。小于该行程说明夹爪几乎没闭合，前方可能有物体或硬件异常。
    gripper_calibration_min_travel: int = 100
    # 运动前如果夹爪不是闭合状态，先收拢到空载基线再开始转移。张开的手指
    # 是比 tool_guard 固定防撞盒更宽、更不可预测的碰撞形状，且可能在长距离
    # MoveIt 转移途中勾挂到东西；闭合是已知、更小的包络。100 远高于实测闭合
    # 基线（pos≈0）的抖动量级，远低于张开位（pos≈900），足以只在真正张开
    # 时触发。--finish-from-current 场景假设夹爪已抓着水瓶，跳过这一步。
    gripper_pretransit_open_threshold: int = 100
    # 空夹基线的静态回退值（2026-07-16 实测 pos~=394）。每次运行会在自由
    # 空间重新实测基线（RobotSession.calibrate_empty_close），静态值只在
    # 没标定成功时兜底。余量取 6：2026-07-15 实测抓稳的窄金属瓶只比空夹
    # 基线高 8，旧余量 35 把真实成功误判成空夹。
    gripper_empty_closed_position: int = 394
    gripper_object_margin: int = 6


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    confidence: float
    class_name: str


@dataclass
class Localization:
    point_camera: list[float]
    point_base: list[float]
    pixel: list[float]
    depth_m: float
    depth_mad_m: float
    position_spread_m: float
    box: list[int]
    confidence: float
    frame_count: int
    class_name: str | None = None
    captured_at_utc: str | None = None


def pose_matrix(pose: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", pose[3:6]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def matrix_pose(T: np.ndarray) -> list[float]:
    return [
        *map(float, T[:3, 3]),
        *map(float, Rotation.from_matrix(T[:3, :3]).as_euler("xyz")),
    ]


def interpolate_poses(
    start: Sequence[float],
    end: Sequence[float],
    max_step: float,
    max_rotation_step_deg: float = 10.0,
) -> list[list[float]]:
    """Interpolate a Cartesian path with translation and rotation step bounds."""
    a, b = pose_matrix(start), pose_matrix(end)
    distance = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
    if max_step <= 0 or max_rotation_step_deg <= 0:
        raise SafetyAbort("位姿插值步长必须为正数")
    relative = Rotation.from_matrix(a[:3, :3].T @ b[:3, :3])
    rotation_deg = float(np.degrees(relative.magnitude()))
    count = max(
        1,
        int(math.ceil(distance / max_step)),
        int(math.ceil(rotation_deg / max_rotation_step_deg)),
    )
    relative_rotvec = relative.as_rotvec()
    result = []
    for i in range(1, count + 1):
        alpha = i / count
        T = np.eye(4)
        T[:3, 3] = (1 - alpha) * a[:3, 3] + alpha * b[:3, 3]
        T[:3, :3] = (
            a[:3, :3]
            @ Rotation.from_rotvec(alpha * relative_rotvec).as_matrix()
        )
        result.append(matrix_pose(T))
    return result


def interpolate_joint_path(
    start_joints_deg: Sequence[float],
    points_deg: Sequence[Sequence[float]],
    max_step_deg: float,
) -> list[list[float]]:
    """Densify a joint path with a hard per-joint angular step bound."""
    if not math.isfinite(max_step_deg) or max_step_deg <= 0:
        raise SafetyAbort(f"关节插值步长无效: {max_step_deg}")
    current = np.asarray(start_joints_deg, dtype=float)
    if current.ndim != 1 or not np.all(np.isfinite(current)):
        raise SafetyAbort("关节插值起点无效")
    dense: list[list[float]] = []
    for index, target_values in enumerate(points_deg, 1):
        target = np.asarray(target_values, dtype=float)
        if target.shape != current.shape or not np.all(np.isfinite(target)):
            raise SafetyAbort(f"关节轨迹点 {index} 维度或数值无效")
        count = max(
            1,
            int(np.ceil(np.max(np.abs(target - current)) / max_step_deg)),
        )
        for step in range(1, count + 1):
            alpha = step / count
            dense.append(((1 - alpha) * current + alpha * target).tolist())
        current = target
    return dense


def look_at_camera_pose(
    target: Sequence[float],
    camera_position: Sequence[float],
) -> np.ndarray:
    """Construct an optical camera pose whose +Z axis looks at the target.

    RealSense optical coordinates use +X right, +Y down, +Z forward. The
    projected base -Z direction is used as camera down to keep the image level.
    """
    target = np.asarray(target, dtype=float)
    position = np.asarray(camera_position, dtype=float)
    z_axis = target - position
    z_axis /= np.linalg.norm(z_axis)
    down = np.array([0.0, 0.0, -1.0])
    down -= z_axis * float(down @ z_axis)
    if np.linalg.norm(down) < 1e-5:
        down = np.array([0.0, 1.0, 0.0])
        down -= z_axis * float(down @ z_axis)
    y_axis = down / np.linalg.norm(down)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4)
    T[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    T[:3, 3] = position
    return T
