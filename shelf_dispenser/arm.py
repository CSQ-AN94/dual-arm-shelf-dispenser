"""RealMan RM75 connection, TCP setup, IK validation, and motion primitives."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .console import ProgressReporter
from .core import (
    DemoParams,
    SafetyAbort,
    interpolate_joint_path,
    matrix_pose,
    pose_matrix,
    stop_reason,
)

LOG = logging.getLogger("bottle_demo")

JOINT_ERROR_NAMES = {
    0x0001: "FOC错误",
    0x0002: "过压",
    0x0004: "欠压",
    0x0008: "过温",
    0x0010: "启动失败",
    0x0020: "编码器错误",
    0x0040: "过流",
    0x0080: "软件错误",
    0x0100: "温度传感器错误",
    0x0200: "位置超限",
    0x0400: "关节ID非法",
    0x0800: "位置跟踪错误",
    0x1000: "电流检测错误",
    0x2000: "抱闸打开失败",
    0x4000: "位置指令阶跃警告",
    0x8000: "多圈关节丢圈数",
    0xF000: "通信丢帧",
}

CONNECTED_TRAJECTORY_MAX_COMMANDS = 30
CONNECTED_TRAJECTORY_MAX_ERROR_DEG = 0.02
CONNECTED_TRAJECTORY_MAX_STEP_DEG = 15.0
# A planned first point within the compressor's own path-error budget of the
# measured start is, by that same standard, the start: fitting it costs a queue
# slot and moves ~0.1 mm.  The 2026-08-02 shelf pick compressed to exactly 30
# commands from the planned start and 31 from any live start, because the
# pre-flight check ran on a snapshot while the executor re-read feedback -- so
# the gripper opened and only then was the path refused.  Dropping that lead-in
# makes both counts identical for drift up to the budget (measured drift was
# 0.004 deg).  Anything larger is the arm having actually moved, and still
# fails loudly.
CONNECTED_TRAJECTORY_START_NOOP_DEG = CONNECTED_TRAJECTORY_MAX_ERROR_DEG
# How long a commanded point may take to settle before its tracking error is
# judged.  Bounded so a genuinely stuck arm still fails, rather than waiting on
# a controller that will never arrive.
TRACKING_SETTLE_TIMEOUT_S = 2.0
TRACKING_SETTLE_POLL_S = 0.05


def validate_open_gripper_feedback(state: dict, params: DemoParams) -> dict:
    """Pure gate shared by RobotSession and the MTC trajectory validator."""
    try:
        pos = int(state["pos"][0])
        dof_state = int(state["dof_state"][0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SafetyAbort("夹爪打开反馈缺失或格式无效") from exc
    if dof_state != 2 or pos < params.gripper_open_position - 50:
        raise SafetyAbort(f"夹爪未可靠打开: state={dof_state}, pos={pos}")
    return state


def validate_holding_gripper_feedback(
    state: dict,
    params: DemoParams,
    *,
    empty_close_pos: int | None = None,
) -> dict:
    """Require settled force-hold plus a gap above the measured empty baseline."""
    try:
        dof_state = int(state["dof_state"][0])
        pos = int(state["pos"][0])
        current = int(state["current"][0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SafetyAbort("夹爪闭合反馈缺失或格式无效") from exc
    baseline = (
        params.gripper_empty_closed_position
        if empty_close_pos is None
        else int(empty_close_pos)
    )
    minimum_object_pos = baseline + params.gripper_object_margin
    if dof_state != 3:
        raise SafetyAbort(
            "夹爪未达到内部夹持力，禁止抬升: "
            f"state={dof_state}, pos={pos}, current={current}"
        )
    if pos <= minimum_object_pos:
        raise SafetyAbort(
            "夹爪闭合位置等同空夹，判定未抓到水瓶，禁止抬升: "
            f"pos={pos}, 空夹基线={baseline}, current={current}。"
            "若现场确认实际已夹稳，记录本行数据后调小 "
            "gripper_object_margin"
        )
    return state


def _tool_transform(
    tcp_z_m: float,
    configured: Sequence[Sequence[float]] | None,
    *,
    label: str = "控制器法兰→TCP",
) -> np.ndarray:
    """Return one named tool-chain segment with rigid validation."""
    if configured is None:
        transform = np.eye(4)
        transform[2, 3] = float(tcp_z_m)
        return transform
    transform = np.asarray(configured, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise SafetyAbort(f"{label} 必须是 4x4 有限刚体变换")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise SafetyAbort(f"{label} 最后一行必须是 [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise SafetyAbort(f"{label} 旋转必须正交")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise SafetyAbort(f"{label} 不允许镜像旋转")
    return transform.copy()


class ArmJointReader:
    """Read another arm in a subprocess to isolate SDK-global install angles."""

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port

    def joints_deg(self) -> list[float]:
        code = (
            "import json\n"
            "from Robotic_Arm.rm_robot_interface import "
            "RoboticArm,rm_thread_mode_e\n"
            f"a=RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)\n"
            f"h=a.rm_create_robot_arm({self.ip!r},{self.port})\n"
            "rc,q=a.rm_get_joint_degree()\n"
            "a.rm_delete_robot_arm()\n"
            "print('BOTTLE_JOINTS_JSON='+json.dumps({'rc':rc,'q':q}))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=8,
        )
        marker = "BOTTLE_JOINTS_JSON="
        line = next(
            (
                item[len(marker) :]
                for item in result.stdout.splitlines()
                if item.startswith(marker)
            ),
            None,
        )
        if result.returncode != 0 or line is None:
            raise SafetyAbort(
                "另一机械臂关节读取子进程失败: "
                f"rc={result.returncode}, stderr={result.stderr.strip()}"
            )
        payload = json.loads(line)
        if payload["rc"] != 0:
            raise SafetyAbort(f"读取另一机械臂关节角失败: {payload['rc']}")
        return list(map(float, payload["q"]))

    def close(self):
        return


class RobotSession:
    def __init__(
        self,
        ip: str,
        port: int,
        stop_event: threading.Event,
        tcp_z_m: float,
        model_flange_offset_m: float = 0.0172,
        take_control: bool = True,
        tcp_transform: Sequence[Sequence[float]] | None = None,
        link7_to_controller_flange: Sequence[Sequence[float]] | None = None,
    ):
        from Robotic_Arm.rm_robot_interface import (
            Algo,
            RoboticArm,
            rm_force_type_e,
            rm_frame_t,
            rm_inverse_kinematics_params_t,
            rm_robot_arm_model_e,
            rm_thread_mode_e,
        )

        self.rm_frame_t = rm_frame_t
        self.ik_params = rm_inverse_kinematics_params_t
        self.stop_event = stop_event
        self.tcp_z_m = tcp_z_m
        self.tcp_transform = _tool_transform(tcp_z_m, tcp_transform)
        self.model_flange_offset_m = model_flange_offset_m
        self.link7_to_controller_flange = _tool_transform(
            model_flange_offset_m,
            link7_to_controller_flange,
            label="r_link7→控制器法兰",
        )
        self.take_control = take_control
        self.closed = False
        if self.take_control:
            self._stop_teleop()
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(ip, port)
        if self.handle.id == -1:
            raise SafetyAbort("机械臂 SDK 连接失败")
        self.algo = Algo(
            rm_robot_arm_model_e.RM_MODEL_RM_75_E,
            rm_force_type_e.RM_MODEL_RM_B_E,
        )
        install = self.arm.rm_get_install_pose()
        if install.get("return_code") != 0:
            raise SafetyAbort(f"读取机械臂安装角失败: {install}")
        install_angles = [
            float(install[key]) for key in ("x", "y", "z")
        ]
        self.algo.rm_algo_set_angle(*install_angles)
        LOG.info("加载机械臂安装角: %s", install_angles)
        self.algo.rm_algo_set_redundant_parameter_traversal_mode(True)
        if self.take_control:
            self.arm.rm_set_tool_voltage(3)
            time.sleep(0.5)
            self.arm.rm_set_rm_plus_mode(115200)
            time.sleep(0.3)
            self.set_tcp()
        self.monitor = None
        if self.take_control:
            self.monitor = threading.Thread(target=self._monitor_stop, daemon=True)
            self.monitor.start()

    @staticmethod
    def _stop_teleop():
        subprocess.run(["pkill", "-x", "atom"], check=False)
        subprocess.run(["pkill", "-f", "zhixing_ctrl.py"], check=False)
        time.sleep(1.5)

    def set_tcp(self):
        expected_transform = self._tcp_transform()
        expected_pose = matrix_pose(expected_transform)
        frame = self.rm_frame_t(
            "bottleTCP", expected_pose, 0, 0, 0, 0
        )
        rc = self.arm.rm_set_manual_tool_frame(frame)
        if rc != 0:
            get_rc, existing = self.arm.rm_get_given_tool_frame("bottleTCP")
            if get_rc != 0:
                raise SafetyAbort(
                    f"设置真实 TCP 失败且无法读取已有坐标系: {rc}/{get_rc}"
                )
            pose = existing.get("pose", [])
            if not self._tool_pose_matches(pose, expected_transform):
                update_rc = self.arm.rm_update_tool_frame(frame)
                if update_rc != 0:
                    raise SafetyAbort(
                        f"更新已有真实 TCP 失败: {update_rc}"
                    )
            LOG.info("复用已有工具坐标 bottleTCP")
        rc = self.arm.rm_change_tool_frame("bottleTCP")
        if rc != 0:
            raise SafetyAbort(f"切换真实 TCP 失败: {rc}")
        # Controller tool frames are flange-relative, while RealMan Algo tool
        # frames are relative to the model endpoint r_link7.
        self._set_algo_tool_transform(
            self.link7_to_controller_flange @ expected_transform
        )

    def _tcp_transform(self) -> np.ndarray:
        """Read a validated full tool transform, with legacy test fallback."""
        configured = getattr(self, "tcp_transform", None)
        return _tool_transform(self.tcp_z_m, configured)

    @staticmethod
    def _tool_pose_matches(
        pose: Sequence[float], expected_transform: np.ndarray
    ) -> bool:
        try:
            actual = pose_matrix(pose)
        except (TypeError, ValueError):
            return False
        if actual.shape != (4, 4) or not np.all(np.isfinite(actual)):
            return False
        position_error = float(
            np.linalg.norm(actual[:3, 3] - expected_transform[:3, 3])
        )
        relative = expected_transform[:3, :3].T @ actual[:3, :3]
        cosine = float((np.trace(relative) - 1.0) / 2.0)
        orientation_error_deg = float(
            np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        )
        return position_error <= 1e-5 and orientation_error_deg <= 0.01

    def _set_algo_tool_transform(self, transform: np.ndarray):
        pose = matrix_pose(
            _tool_transform(
                self.tcp_z_m, transform, label="r_link7→TCP"
            )
        )
        frame = self.rm_frame_t("algoTool", pose, 0, 0, 0, 0)
        self.algo.rm_algo_set_toolframe(frame)

    def _set_algo_tool_z(self, z_m: float):
        # Keep this compatibility seam for offline tests and legacy call
        # sites. RealMan Algo's zero-tool FK is the model's r_link7 pose, not
        # the controller flange. Therefore both branches must start with the
        # measured r_link7→controller-flange segment.
        link7_to_flange = np.asarray(
            self.link7_to_controller_flange, dtype=float
        )
        if np.isclose(float(z_m), float(self.tcp_z_m), atol=1e-12):
            self._set_algo_tool_transform(
                link7_to_flange @ self._tcp_transform()
            )
            return
        offset = np.eye(4)
        offset[2, 3] = float(z_m)
        self._set_algo_tool_transform(link7_to_flange @ offset)

    def controller_flange_from_joints(
        self, joints_deg: Sequence[float]
    ) -> np.ndarray:
        # RealMan Algo's zero tool resolves to the kinematic model endpoint
        # r_link7. Activate the separately calibrated controller-flange
        # segment before asking for flange FK.
        self._set_algo_tool_z(0.0)
        pose = self.algo.rm_algo_forward_kinematics(
            list(map(float, joints_deg)), 1
        )
        if not pose or len(pose) != 6:
            raise SafetyAbort("SDK 正解未返回有效控制器法兰位姿")
        return pose_matrix(pose)

    def _monitor_stop(self):
        self.stop_event.wait()
        if not self.closed:
            try:
                self.arm.rm_set_arm_slow_stop()
            except Exception:
                LOG.exception("缓停命令失败；请使用硬件急停")

    def current_tcp(self) -> np.ndarray:
        if not self.take_control:
            raise SafetyAbort("只规划会话未设置 TCP，禁止读取 TCP 位姿")
        code, state = self.arm.rm_get_current_arm_state()
        if code != 0 or not state.get("pose"):
            raise SafetyAbort(f"读取 TCP 位姿失败: {code}")
        for key in ("arm_err", "sys_err"):
            value = state.get(key, 0)
            if value not in (0, None, [0]):
                raise SafetyAbort(f"控制器报告 {key}={value}")
        nested = state.get("err")
        if isinstance(nested, dict):
            values = nested.get("err", [])
            if any(str(value) != "0" for value in values):
                raise SafetyAbort(f"控制器报告 err={nested}")
        return pose_matrix(state["pose"])

    def assert_arm_healthy(self) -> dict:
        """Fail closed on joint-level faults hidden by the summary state.

        `rm_get_current_arm_state()` can report aggregate err=0 while
        `rm_get_arm_all_state()` still carries a live per-joint error.  A real
        J7 0xF000 frame-loss fault produced API2 -6 for every motion command.
        """
        rc, state = self.arm.rm_get_arm_all_state()
        if rc != 0:
            raise SafetyAbort(f"读取机械臂完整关节状态失败: rc={rc}")
        enabled = list(state.get("joint_en_flag", []))
        errors = [int(value) for value in state.get("joint_err_code", [])]
        if len(enabled) != 7 or len(errors) != 7:
            raise SafetyAbort(f"机械臂完整关节状态字段不完整: {state}")
        disabled = [f"J{index}" for index, value in enumerate(enabled, 1) if not value]
        faults = []
        for index, code in enumerate(errors, 1):
            if code:
                name = JOINT_ERROR_NAMES.get(code, "未知关节错误")
                faults.append(f"J{index}=0x{code:04X}({name})")
        if disabled or faults:
            detail = []
            if disabled:
                detail.append("未使能=" + ",".join(disabled))
            if faults:
                detail.append("关节错误=" + ",".join(faults))
            raise SafetyAbort(
                "机械臂关节级自检失败: " + "; ".join(detail)
            )

        controller = self.arm.rm_get_controller_state()
        if controller.get("return_code") != 0:
            raise SafetyAbort(f"读取控制器状态失败: {controller}")
        system_error = int(
            controller.get("system_error", controller.get("sys_err", 0))
        )
        if system_error:
            raise SafetyAbort(
                f"机械臂控制器系统错误=0x{system_error:04X}: {controller}"
            )
        return {"joints": state, "controller": controller}

    def recover_transient_joint_frame_loss(self) -> list[int]:
        """Clear a stale 0xF000 joint flag once, under narrow safe conditions.

        Holding the end-effector green drag button can leave a joint's
        communication-frame-loss flag latched after the button is released.
        The controller otherwise remains healthy and all joints stay enabled.
        Only that exact state is auto-cleared.  Every other error remains a
        hard preflight failure, and 0xF000 must stay clear on two subsequent
        reads before motion is allowed.
        """
        rc, state = self.arm.rm_get_arm_all_state()
        if rc != 0:
            return []
        enabled = [int(value) for value in state.get("joint_en_flag", [])]
        errors = [int(value) for value in state.get("joint_err_code", [])]
        affected = [
            index for index, code in enumerate(errors, 1) if code == 0xF000
        ]
        nonzero = [code for code in errors if code]
        if (
            not affected
            or len(enabled) != 7
            or len(errors) != 7
            or any(value != 1 for value in enabled)
            or any(code != 0xF000 for code in nonzero)
        ):
            return []

        controller = self.arm.rm_get_controller_state()
        if controller.get("return_code") != 0:
            return []
        system_error = int(
            controller.get("system_error", controller.get("sys_err", 0))
        )
        if system_error:
            return []

        LOG.warning(
            "检测到仅有的瞬态关节通信丢帧标志 %s；执行一次官方错误清除并复核",
            affected,
        )
        for joint in affected:
            clear_rc = self.arm.rm_set_joint_clear_err(joint)
            if clear_rc != 0:
                raise SafetyAbort(
                    f"清除 J{joint} 瞬态通信丢帧失败: rc={clear_rc}"
                )

        for verification in range(1, 3):
            time.sleep(0.5)
            verify_rc, verify_state = self.arm.rm_get_arm_all_state()
            verify_errors = [
                int(value) for value in verify_state.get("joint_err_code", [])
            ]
            if verify_rc != 0 or len(verify_errors) != 7:
                raise SafetyAbort(
                    "清除瞬态通信丢帧后无法读取完整关节状态"
                )
            if any(verify_errors):
                raise SafetyAbort(
                    "关节通信丢帧清除后再次出现，拒绝运动: "
                    f"第 {verification} 次复核 errors={verify_errors}"
                )
        return affected

    def current_flange(self) -> np.ndarray:
        if not self.take_control:
            # A visual-only resume check must not change the active controller
            # tool frame merely to read the wrist-camera transform.  Joint FK
            # gives the flange pose without stopping teleop, changing voltage,
            # or clearing any controller fault.
            return self.controller_flange_from_joints(self.joints_deg())
        T_tcp = self.current_tcp()
        return T_tcp @ np.linalg.inv(self._tcp_transform())

    def joints_deg(self) -> list[float]:
        rc, joints = self.arm.rm_get_joint_degree()
        if rc != 0:
            raise SafetyAbort(f"读取关节角失败: {rc}")
        return list(map(float, joints))

    def plan_ik(
        self,
        poses: Sequence[Sequence[float]],
        params: DemoParams,
        *,
        allow_first_jump: bool = False,
        seed_joints_deg: Sequence[float] | None = None,
    ) -> list[list[float]]:
        """Solve a sequential IK chain with limit/singularity/jump guards.

        seed_joints_deg lets callers check a hypothetical path from a pose the
        arm is *not* currently in (e.g. grasp-feasibility precheck of an
        observation candidate before ever moving there). No motion happens
        here either way.
        """
        self._set_algo_tool_z(self.tcp_z_m)
        q = (
            list(map(float, seed_joints_deg))
            if seed_joints_deg is not None
            else self.joints_deg()
        )
        rc_min, qmin = self.arm.rm_get_joint_min_pos()
        rc_max, qmax = self.arm.rm_get_joint_max_pos()
        if rc_min != 0 or rc_max != 0:
            raise SafetyAbort("无法读取控制器关节限位")
        self.algo.rm_algo_set_joint_min_limit(list(qmin))
        self.algo.rm_algo_set_joint_max_limit(list(qmax))
        start_in_band = abs(q[3]) < params.j4_singularity_deg
        planned = []
        for idx, pose in enumerate(poses):
            rc, solution = self.algo.rm_algo_inverse_kinematics(
                self.ik_params(q, list(pose), 1)
            )
            if rc != 0:
                raise SafetyAbort(f"路径点 {idx + 1}/{len(poses)} 逆解失败: {rc}")
            solution = list(map(float, solution))
            for joint, (angle, lo, hi) in enumerate(
                zip(solution, qmin, qmax), 1
            ):
                if not (
                    lo + params.joint_limit_margin_deg
                    <= angle
                    <= hi - params.joint_limit_margin_deg
                ):
                    raise SafetyAbort(
                        f"路径点 {idx + 1} 关节 J{joint} 距限位过近: {angle:.1f}°"
                    )
            if abs(solution[3]) < params.j4_singularity_deg:
                hint = (
                    "；起点姿态本身已在奇异带内，需先做关节空间弯肘逃逸"
                    if start_in_band
                    else "；目标可能接近手臂最大伸展，需调整目标或移动底盘"
                )
                raise SafetyAbort(
                    f"路径点 {idx + 1} J4={solution[3]:.1f}°，进入奇异区{hint}"
                )
            if (
                not (allow_first_jump and idx == 0)
                and max(abs(a - b) for a, b in zip(solution, q)) > 28
            ):
                raise SafetyAbort(f"路径点 {idx + 1} 逆解关节跳变超过 28°")
            planned.append(solution)
            q = solution
        return planned

    def escape_j4_singularity(self, params, safety_profile) -> list[float] | None:
        """若当前姿态在 J4≈0 奇异带内，用关节空间弯肘运动先离开该带。

        肘角大小由肩-腕距离唯一决定，任何保持 TCP 位姿不变的重试（包括
        绕工具 z 轴 roll）都改不了 |J4|。而对控制器来说，纯关节运动不经过
        病态雅可比，穿越/离开奇异带是安全的——危险的只是在带内做笛卡尔
        直线。因此这里只动 J4，逃逸路径逐点做 FK+电子围栏校验后用 movej
        执行。返回逃逸后的目标关节角；不在带内则返回 None。
        """
        q = self.joints_deg()
        if abs(q[3]) >= params.j4_singularity_deg:
            return None
        if not self.take_control:
            raise SafetyAbort(
                f"当前 J4={q[3]:.1f}° 在奇异带内，只规划会话无法执行弯肘逃逸"
            )
        rc_min, qmin = self.arm.rm_get_joint_min_pos()
        rc_max, qmax = self.arm.rm_get_joint_max_pos()
        if rc_min != 0 or rc_max != 0:
            raise SafetyAbort("无法读取控制器关节限位")
        preferred = 1.0 if q[3] >= 0 else -1.0
        rejections = []
        for sign in (preferred, -preferred):
            target = list(q)
            target[3] = sign * params.j4_escape_deg
            lo = qmin[3] + params.joint_limit_margin_deg
            hi = qmax[3] - params.joint_limit_margin_deg
            if not (lo <= target[3] <= hi):
                rejections.append(f"J4={target[3]:.1f}° 距限位过近")
                continue
            dense = interpolate_joint_path(
                q, [target], params.planned_joint_step_deg
            )
            try:
                for index, joints in enumerate(dense, 1):
                    tcp = self.tcp_from_joints(joints)
                    safety_profile.assert_tcp_point(
                        tcp[:3, 3], label=f"J4 逃逸路径点 {index}"
                    )
            except SafetyAbort as exc:
                rejections.append(str(exc))
                continue
            LOG.info(
                "J4 奇异带逃逸: %.1f° -> %.1f°，%d 个围栏校验点通过",
                q[3],
                target[3],
                len(dense),
            )
            self.execute_planned_joints(
                [target], params.final_speed, params.planned_joint_step_deg
            )
            return target
        raise SafetyAbort(
            f"当前 J4={q[3]:.1f}° 在奇异带内，两个弯肘方向均不可行: "
            + "；".join(rejections)
        )

    @staticmethod
    def _equivalent_joint_seeds(
        seed_joints_deg: Sequence[float],
        qmin: Sequence[float],
        qmax: Sequence[float],
        margin_deg: float,
    ) -> list[list[float]]:
        """Return bounded +/-360deg representations of one joint state.

        RM75's first six joints cannot contain another full turn inside their
        controller limits.  J7 is different (normally +/-360deg): the same
        physical wrist orientation can therefore be represented on two turns.
        Try those representations as IK seeds, but never invent an unbounded
        continuous joint or combine several full-turn changes at once.
        """
        seed = np.asarray(seed_joints_deg, dtype=float)
        lower = np.asarray(qmin, dtype=float)
        upper = np.asarray(qmax, dtype=float)
        if (
            seed.shape != (7,)
            or lower.shape != (7,)
            or upper.shape != (7,)
            or not np.all(np.isfinite(seed))
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
        ):
            raise SafetyAbort("关节换圈种子或控制器限位无效")
        variants = [seed.tolist()]
        for joint in range(7):
            # A joint whose full usable range is less than one turn cannot
            # have a second equivalent representation inside its limits.
            if upper[joint] - lower[joint] < 360.0 + 2.0 * margin_deg:
                continue
            for delta in (-360.0, 360.0):
                candidate = seed.copy()
                candidate[joint] += delta
                if (
                    lower[joint] + margin_deg
                    <= candidate[joint]
                    <= upper[joint] - margin_deg
                ):
                    variants.append(candidate.tolist())
        return variants

    def solve_flange_ik_candidates(
        self,
        target_controller_flange: np.ndarray,
        params: DemoParams,
        seed_joints_deg: Sequence[float] | None = None,
    ) -> list[list[float]]:
        """Solve bounded J7 turns and a small, diverse set of RM75 elbow branches."""
        self._set_algo_tool_z(0.0)
        seed = np.asarray(
            (
                self.joints_deg()
                if seed_joints_deg is None
                else seed_joints_deg
            ),
            dtype=float,
        )
        if seed.shape != (7,) or not np.all(np.isfinite(seed)):
            raise SafetyAbort("控制器同源法兰逆解种子必须是 7 个有限关节角")
        rc_min, qmin = self.arm.rm_get_joint_min_pos()
        rc_max, qmax = self.arm.rm_get_joint_max_pos()
        if rc_min != 0 or rc_max != 0:
            raise SafetyAbort("无法读取控制器关节限位")
        self.algo.rm_algo_set_joint_min_limit(list(qmin))
        self.algo.rm_algo_set_joint_max_limit(list(qmax))
        solutions: list[list[float]] = []
        rejections: list[str] = []
        target_pose = matrix_pose(
            np.asarray(target_controller_flange, dtype=float)
        )

        def add_solution(raw_solution: Sequence[float]) -> bool:
            solution = list(map(float, raw_solution))
            if len(solution) != 7 or not np.all(np.isfinite(solution)):
                rejections.append("逆解返回的关节数组无效")
                return False
            invalid = next(
                (
                    (joint, angle)
                    for joint, (angle, lo, hi) in enumerate(
                        zip(solution, qmin, qmax), 1
                    )
                    if not (
                        lo + params.joint_limit_margin_deg
                        <= angle
                        <= hi - params.joint_limit_margin_deg
                    )
                ),
                None,
            )
            if invalid is not None:
                joint, angle = invalid
                rejections.append(f"J{joint}={angle:.1f}° 距限位过近")
                return False
            if abs(solution[3]) < params.j4_singularity_deg:
                rejections.append(f"J4={solution[3]:.1f}° 进入奇异区")
                return False
            if any(
                np.allclose(solution, existing, atol=1e-4, rtol=0.0)
                for existing in solutions
            ):
                return False
            solutions.append(solution)
            return True

        for ik_seed in self._equivalent_joint_seeds(
            seed,
            qmin,
            qmax,
            params.joint_limit_margin_deg,
        ):
            rc, raw_solution = self.algo.rm_algo_inverse_kinematics(
                self.ik_params(
                    ik_seed,
                    target_pose,
                    1,
                )
            )
            if rc != 0:
                rejections.append(f"seed J7={ik_seed[6]:.1f}° rc={rc}")
                continue
            add_solution(raw_solution)

        # Traversal mode still returns only one RM75 redundancy choice.  The
        # model-specific arm-angle API holds the flange fixed while changing
        # how the elbow circles the shoulder-wrist axis.  Sample near the
        # current arm angle first and cap the result: endpoint/continuation
        # collision validation remains the authoritative selector, but must
        # not explode into hundreds of MoveIt calls.
        calculate_arm_angle = getattr(
            self.algo, "rm_algo_calculate_arm_angle_from_config_rm75", None
        )
        solve_arm_angle = getattr(
            self.algo, "rm_algo_inverse_kinematics_rm75_for_arm_angle", None
        )
        max_candidates = 4
        if callable(calculate_arm_angle) and callable(solve_arm_angle):
            angle_rc, current_arm_angle = calculate_arm_angle(seed.tolist())
            if angle_rc == 0 and np.isfinite(float(current_arm_angle)):
                angle_candidates: list[float] = []
                for offset in (
                    0.0,
                    30.0,
                    -30.0,
                    60.0,
                    -60.0,
                    90.0,
                    -90.0,
                    120.0,
                    -120.0,
                    150.0,
                    -150.0,
                    180.0,
                ):
                    angle = (
                        (float(current_arm_angle) + offset + 180.0) % 360.0
                    ) - 180.0
                    if not any(
                        abs(angle - existing) < 1e-6
                        for existing in angle_candidates
                    ):
                        angle_candidates.append(angle)
                arm_params = self.ik_params(
                    seed.tolist(), target_pose, 1
                )
                for arm_angle in angle_candidates:
                    if len(solutions) >= max_candidates:
                        break
                    rc, raw_solution = solve_arm_angle(
                        arm_params, arm_angle
                    )
                    if rc != 0:
                        continue
                    for equivalent in self._equivalent_joint_seeds(
                        raw_solution,
                        qmin,
                        qmax,
                        params.joint_limit_margin_deg,
                    ):
                        add_solution(equivalent)
                        if len(solutions) >= max_candidates:
                            break
        if not solutions:
            detail = "；".join(rejections) or "控制器未返回候选"
            raise SafetyAbort(f"控制器同源法兰逆解全部失败: {detail}")
        return solutions

    def solve_flange_ik(
        self,
        target_controller_flange: np.ndarray,
        params: DemoParams,
        seed_joints_deg: Sequence[float] | None = None,
    ) -> list[float]:
        """Compatibility helper: return the current-turn IK branch first."""
        return self.solve_flange_ik_candidates(
            target_controller_flange,
            params,
            seed_joints_deg=seed_joints_deg,
        )[0]

    @staticmethod
    def _dense_joint_path(
        start_joints_deg: Sequence[float],
        points_deg: Sequence[Sequence[float]],
        max_step_deg: float,
    ) -> list[list[float]]:
        return interpolate_joint_path(
            start_joints_deg, points_deg, max_step_deg
        )

    @staticmethod
    def _compress_connected_joint_path(
        start_joints_deg: Sequence[float],
        points_deg: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """Fit the checked polyline into the controller's 30-command queue."""
        path = [np.asarray(start_joints_deg, dtype=float)] + [
            np.asarray(point, dtype=float) for point in points_deg
        ]
        if any(point.shape != (7,) or not np.all(np.isfinite(point)) for point in path):
            raise SafetyAbort("连续轨迹压缩输入含非有限数或维度无效")
        # Every originally planned segment is bounded before anything is
        # dropped, so a lead-in can never hide an oversized step behind it.
        if any(
            float(np.max(np.abs(after - before)))
            > CONNECTED_TRAJECTORY_MAX_STEP_DEG
            for before, after in zip(path, path[1:])
        ):
            raise SafetyAbort(
                "连续轨迹相邻原始点超过控制器单段上限 "
                f"{CONNECTED_TRAJECTORY_MAX_STEP_DEG:.1f}°"
            )
        if (
            len(path) > 2
            and float(np.max(np.abs(path[1] - path[0])))
            <= CONNECTED_TRAJECTORY_START_NOOP_DEG
            and float(np.max(np.abs(path[2] - path[0])))
            <= CONNECTED_TRAJECTORY_MAX_STEP_DEG
        ):
            del path[1]

        commands: list[list[float]] = []
        anchor = 0
        while anchor < len(path) - 1:
            farthest = anchor + 1
            for end in range(anchor + 2, len(path)):
                direction = path[end] - path[anchor]
                length_squared = float(direction @ direction)
                if (
                    length_squared <= 1e-12
                    or float(np.max(np.abs(direction)))
                    > CONNECTED_TRAJECTORY_MAX_STEP_DEG
                ):
                    break
                previous_fraction = -1.0
                fits = True
                for point in path[anchor + 1 : end]:
                    fraction = float(
                        (point - path[anchor]) @ direction / length_squared
                    )
                    if (
                        fraction < 0.0
                        or fraction > 1.0
                        or fraction < previous_fraction
                        or float(
                            np.max(
                                np.abs(
                                    point
                                    - (path[anchor] + fraction * direction)
                                )
                            )
                        )
                        > CONNECTED_TRAJECTORY_MAX_ERROR_DEG
                    ):
                        fits = False
                        break
                    previous_fraction = fraction
                if not fits:
                    break
                farthest = end
            commands.append(path[farthest].tolist())
            anchor = farthest

        if len(commands) > CONNECTED_TRAJECTORY_MAX_COMMANDS:
            raise SafetyAbort(
                "连续轨迹在 0.02° 最大关节误差、15° 单段上限下仍需 "
                f"{len(commands)} 个控制点，超过控制器队列上限 "
                f"{CONNECTED_TRAJECTORY_MAX_COMMANDS}；拒绝放宽路径误差"
            )
        return commands

    def tcp_from_joints(self, joints_deg: Sequence[float]) -> np.ndarray:
        self._set_algo_tool_z(self.tcp_z_m)
        tcp_pose = self.algo.rm_algo_forward_kinematics(
            list(map(float, joints_deg)), 1
        )
        if not tcp_pose or len(tcp_pose) != 6:
            raise SafetyAbort("SDK 正解未返回有效 TCP 位姿")
        return pose_matrix(tcp_pose)

    def validate_planned_joints(
        self,
        points_deg: Sequence[Sequence[float]],
        max_step_deg: float,
        safety_profile,
        start_joints_deg: Sequence[float] | None = None,
        joint_limit_margin_deg: float = 3.0,
    ) -> int:
        if not points_deg:
            raise SafetyAbort("规划轨迹为空")
        measured_start = (
            self.joints_deg() if start_joints_deg is None else start_joints_deg
        )
        dense = self._dense_joint_path(
            measured_start,
            points_deg,
            max_step_deg,
        )
        rc_min, lower = self.arm.rm_get_joint_min_pos()
        rc_max, upper = self.arm.rm_get_joint_max_pos()
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if (
            rc_min != 0
            or rc_max != 0
            or lower.shape != (7,)
            or upper.shape != (7,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or not np.isfinite(joint_limit_margin_deg)
            or joint_limit_margin_deg <= 0.0
        ):
            raise SafetyAbort("无法验证控制器关节限位余量")
        def _margin_excess(values: np.ndarray) -> np.ndarray:
            """How far past the safety margin each joint sits; never negative."""
            return np.maximum(
                np.maximum(
                    (lower + joint_limit_margin_deg) - values,
                    values - (upper - joint_limit_margin_deg),
                ),
                0.0,
            )

        # The path is judged relative to where the arm already is.  Judging its
        # opening points by the absolute margin deadlocks the robot: on
        # 2026-08-03 the arm sat at J5=175.32 deg against a 175.00 deg margin,
        # four planners each found a way out, and every one was refused on
        # point 1 of 57 -- a point still essentially at the arm's own position.
        # No trajectory can ever be accepted then, because they all start
        # there.
        #
        # A margin exists to stop a plan driving a joint toward its limit, so
        # judge the trajectory by where it goes: a joint may stay as far past
        # the margin as it already is, and may never go further.  A joint that
        # starts inside the margin still may not leave it.
        start = np.asarray(measured_start, dtype=float)
        start_excess = _margin_excess(start)
        for index, joints in enumerate(dense, 1):
            values = np.asarray(joints, dtype=float)
            bad = np.flatnonzero(_margin_excess(values) > start_excess + 1e-9)
            if bad.size:
                joint = int(bad[0])
                raise SafetyAbort(
                    "规划轨迹把关节推向控制器限位: "
                    f"点 {index}/{len(dense)} J{joint + 1}={values[joint]:.2f}°，"
                    f"安全范围=[{lower[joint] + joint_limit_margin_deg:.2f}, "
                    f"{upper[joint] - joint_limit_margin_deg:.2f}]°，"
                    f"起点={start[joint]:.2f}°"
                )
        tcp_points = [self.tcp_from_joints(joints)[:3, 3] for joints in dense]
        return safety_profile.assert_tcp_path(tcp_points)

    def move_linear(
        self,
        pose: Sequence[float],
        speed: int,
        *,
        position_tolerance_m: float = 0.008,
        orientation_tolerance_deg: float = 4.0,
    ):
        """Execute one blocking Cartesian leg and verify measured arrival.

        A zero SDK return code only says that the command completed.  Every
        local grasp/place leg therefore closes the loop with fresh controller
        state before the next waypoint is allowed to run.
        """
        if self.stop_event.is_set():
            raise SafetyAbort(stop_reason(self.stop_event))
        target = list(map(float, pose))
        LOG.info(
            "SDK movel 下发: target=%s speed=%d%%",
            np.round(target, 5).tolist(),
            speed,
        )
        rc = self.arm.rm_movel(target, speed, 0, 0, 1)
        if rc != 0:
            context = self._motion_failure_context(target, speed)
            if rc == -6:
                raise SafetyAbort(
                    "movel 被外部停止指令中止（API2 -6，不是点云障碍或"
                    f"Python 电子围栏拒绝）: {context}"
                )
            raise SafetyAbort(f"movel 失败: rc={rc}; {context}")
        self.assert_arm_healthy()
        actual = self.current_tcp()
        expected = pose_matrix(target)
        position_error = float(
            np.linalg.norm(actual[:3, 3] - expected[:3, 3])
        )
        orientation_error = float(
            np.degrees(
                Rotation.from_matrix(
                    actual[:3, :3].T @ expected[:3, :3]
                ).magnitude()
            )
        )
        if (
            not np.isfinite(position_error)
            or not np.isfinite(orientation_error)
            or position_error > position_tolerance_m
            or orientation_error > orientation_tolerance_deg
        ):
            # A completed-but-missed move is just as unsafe as a failed one:
            # downstream collision checks and grasp state would otherwise be
            # based on a pose that the robot never reached.
            self.hold()
            raise SafetyAbort(
                "movel 执行反馈偏差过大，已停止继续下发: "
                f"位置差={position_error * 1000:.1f} mm "
                f"(上限 {position_tolerance_m * 1000:.1f} mm)，"
                f"姿态差={orientation_error:.1f}° "
                f"(上限 {orientation_tolerance_deg:.1f}°)"
            )

    def _motion_failure_context(
        self, target: Sequence[float], speed: int
    ) -> str:
        """Best-effort snapshot before the SDK connection is torn down."""

        def query(name: str):
            method = getattr(self.arm, name, None)
            if method is None:
                return "unsupported"
            try:
                return method()
            except Exception as exc:  # diagnostics must not hide the first fault
                return f"query_failed:{type(exc).__name__}:{exc}"

        try:
            self.assert_arm_healthy()
            joint_health = "healthy"
        except Exception as exc:  # preserve a decoded joint-level root cause
            joint_health = f"{type(exc).__name__}:{exc}"

        source = getattr(self.stop_event, "source", None)
        context = {
            "target": np.round(np.asarray(target, dtype=float), 5).tolist(),
            "speed": int(speed),
            "local_stop_event": self.stop_event.is_set(),
            "local_stop_source": source,
            "joint_health": joint_health,
            "arm_state": query("rm_get_current_arm_state"),
            "arm_all_state": query("rm_get_arm_all_state"),
            "controller_state": query("rm_get_controller_state"),
            "collision_stage": query("rm_get_collision_stage"),
            "self_collision": query("rm_get_self_collision_enable"),
            "joints": query("rm_get_joint_degree"),
            "controller_fence_enable": query(
                "rm_get_electronic_fence_enable"
            ),
            "controller_fence_config": query(
                "rm_get_electronic_fence_config"
            ),
            "controller_virtual_wall_enable": query(
                "rm_get_virtual_wall_enable"
            ),
        }
        return "; ".join(f"{key}={value}" for key, value in context.items())

    def controller_fence_status(self) -> dict:
        """Read persistent controller-native fence state without changing it."""
        method = getattr(self.arm, "rm_get_electronic_fence_enable", None)
        if method is None:
            raise SafetyAbort("当前 SDK 不支持查询控制器原生电子围栏状态")
        try:
            rc, state = method()
        except Exception as exc:
            raise SafetyAbort(f"查询控制器原生电子围栏状态失败: {exc}") from exc
        if rc != 0:
            raise SafetyAbort(f"查询控制器原生电子围栏状态失败: rc={rc}")

        current = None
        current_method = getattr(
            self.arm, "rm_get_electronic_fence_config", None
        )
        if current_method is not None:
            try:
                current = current_method()
            except Exception as exc:
                current = f"query_failed:{type(exc).__name__}:{exc}"

        saved = None
        list_method = getattr(
            self.arm, "rm_get_electronic_fence_list_infos", None
        )
        if list_method is not None:
            try:
                saved = list_method()
            except Exception as exc:
                saved = f"query_failed:{type(exc).__name__}:{exc}"
        return {"state": state, "current": current, "saved": saved}

    def execute_planned_joints(
        self,
        points_deg: Sequence[Sequence[float]],
        speed: int,
        max_step_deg: float,
        *,
        expected_start_joints_deg: Sequence[float] | None = None,
        start_tolerance_deg: float = 0.8,
        tracking_tolerance_deg: float = 1.2,
    ) -> None:
        """Execute a collision-checked MoveIt path as one connected trajectory.

        MoveIt remains planning-only.  Dense interpolation is still used by
        the validation layer, but sending each dense sample as a separate
        blocking ``rm_movej`` makes the controller brake and restart hundreds
        of times.  Queue the original MoveIt waypoints with RealMan's native
        trajectory-connect flag and execute them as one blended motion.  Set
        ``BOTTLE_GRASP_CONTINUOUS_TRAJECTORY=0`` for the audited blocking
        fallback if a controller firmware rejects connected trajectories.
        """
        if not self.take_control:
            raise SafetyAbort("只规划会话禁止执行运动")
        if not points_deg:
            raise SafetyAbort("规划轨迹为空")
        if (
            not np.isfinite(start_tolerance_deg)
            or start_tolerance_deg <= 0
            or not np.isfinite(tracking_tolerance_deg)
            or tracking_tolerance_deg <= 0
        ):
            raise SafetyAbort("轨迹执行反馈容差必须是正的有限数")

        # Planning and fresh-scene validation can take minutes.  A joint fault
        # may therefore appear after task preflight but before the first real
        # command (2026-07-21: J7 0xF000, followed by API2 -6 at point 1).
        # Re-run the narrowly-scoped transient recovery at the execution
        # boundary, then require a completely healthy arm before any movej.
        recovered = self.recover_transient_joint_frame_loss()
        if recovered:
            LOG.warning(
                "轨迹执行前清除并复核通过瞬态关节通信丢帧: %s",
                ",".join(f"J{joint}" for joint in recovered),
            )
        self.assert_arm_healthy()

        actual_start = np.asarray(self.joints_deg(), dtype=float)
        if actual_start.shape != (7,) or not np.all(np.isfinite(actual_start)):
            raise SafetyAbort("实机规划起点关节反馈含非有限数或维度无效")
        if expected_start_joints_deg is not None:
            expected_start = np.asarray(expected_start_joints_deg, dtype=float)
            if (
                expected_start.shape != actual_start.shape
                or not np.all(np.isfinite(expected_start))
            ):
                raise SafetyAbort("规划起点关节维度或数值与实机不一致")
            start_error = float(np.max(np.abs(actual_start - expected_start)))
            if start_error > start_tolerance_deg:
                raise SafetyAbort(
                    "轨迹已过期：实机已偏离规划起点，拒绝执行: "
                    f"最大关节差={start_error:.2f}°，上限={start_tolerance_deg:.2f}°"
                )
        dense = self._dense_joint_path(actual_start, points_deg, max_step_deg)
        # MoveIt trajectories include their start state.  Sending that state
        # back as a blocking movej creates a useless zero-displacement command
        # and made the real failure misleadingly appear at "point 1".  Drop
        # only feedback-resolution no-ops; accumulated motion is preserved.
        filtered_dense: list[list[float]] = []
        last_command = actual_start
        for joints in dense:
            candidate = np.asarray(joints, dtype=float)
            if float(np.max(np.abs(candidate - last_command))) <= 0.01:
                continue
            filtered_dense.append(candidate.tolist())
            last_command = candidate
        dense = filtered_dense
        if not dense:
            LOG.info("SDK MoveIt 轨迹仅含当前起点，无需下发运动命令")
            return

        continuous = os.environ.get(
            "BOTTLE_GRASP_CONTINUOUS_TRAJECTORY", "1"
        ) != "0"
        execution_points = dense
        if continuous:
            # Preserve MoveIt's path shape while avoiding validation-only
            # interpolation samples as physical stop points.  The already-run
            # dense safety validation still bounds every segment.
            original_execution_points = []
            last_command = actual_start
            for joints in points_deg:
                candidate = np.asarray(joints, dtype=float)
                if candidate.shape != (7,) or not np.all(np.isfinite(candidate)):
                    raise SafetyAbort("MoveIt 轨迹点含非有限数或维度无效")
                # Keep every non-identical planned point: the 0.02°
                # compression proof must not start from the coarser 0.01°
                # feedback no-op filter used by the blocking fallback.
                if float(np.max(np.abs(candidate - last_command))) <= 1e-9:
                    continue
                original_execution_points.append(candidate.tolist())
                last_command = candidate
            if not original_execution_points:
                LOG.info("SDK MoveIt 轨迹仅含当前起点，无需下发运动命令")
                return
            execution_points = self._compress_connected_joint_path(
                actual_start, original_execution_points
            )

        LOG.info(
            "SDK 执行 MoveIt 轨迹: %d 个控制点（原始 %d 点，"
            "安全复核 %d 个密集点），"
            "速度 %d%%，模式=%s",
            len(execution_points),
            len(original_execution_points) if continuous else len(dense),
            len(dense),
            speed,
            "控制器连续交融" if continuous else "阻塞路点回退",
        )
        progress = ProgressReporter("轨迹执行", len(execution_points), logger=LOG)
        for index, joints in enumerate(execution_points, 1):
            if self.stop_event.is_set():
                reason = stop_reason(self.stop_event)
                progress.close(reason)
                clear = getattr(self.arm, "rm_set_delete_current_trajectory", None)
                if callable(clear):
                    clear()
                raise SafetyAbort(reason)
            progress.update(index - 1)
            is_last = index == len(execution_points)
            # Keep the requested connected motion while minimizing controller
            # corner cutting around the already collision-checked waypoints.
            radius = 0 if is_last or not continuous else 1
            connect = 0 if is_last or not continuous else 1
            block = 1 if is_last or not continuous else 0
            rc = self.arm.rm_movej(joints, speed, radius, connect, block)
            if rc != 0:
                progress.close(f"第 {index} 点失败")
                clear = getattr(self.arm, "rm_set_delete_current_trajectory", None)
                if callable(clear):
                    clear()
                context = self._motion_failure_context(joints, speed)
                if rc == -6:
                    raise SafetyAbort(
                        "MoveIt 轨迹被外部停止指令中止（API2 -6，"
                        "不是 MoveIt/点云碰撞复核失败）: "
                        f"点 {index}/{len(execution_points)}; {context}"
                    )
                raise SafetyAbort(
                    f"MoveIt 轨迹点 {index}/{len(execution_points)} 执行失败: "
                    f"rc={rc}; {context}"
                )
            # Connected intermediate commands are queued, not executed yet;
            # only the final blocking command has meaningful arrival feedback.
            if continuous and not is_last:
                continue
            self.current_tcp()
            target = np.asarray(joints, dtype=float)
            # A blended command returns once the controller has accepted it,
            # not once the arm has settled on it.  Sampling immediately reads
            # the arm mid-convergence: on 2026-08-03 a move that finished
            # 0.09 deg from its goal was refused at 1.24 deg against a 1.20 deg
            # limit, measured the instant the last command was acknowledged.
            # Poll until the arm converges, and only then judge it.
            deadline = time.monotonic() + TRACKING_SETTLE_TIMEOUT_S
            while True:
                feedback = np.asarray(self.joints_deg(), dtype=float)
                if feedback.shape != (7,) or not np.all(np.isfinite(feedback)):
                    break
                tracking_error = float(np.max(np.abs(feedback - target)))
                if (
                    tracking_error <= tracking_tolerance_deg
                    or time.monotonic() >= deadline
                    or self.stop_event.is_set()
                ):
                    break
                time.sleep(TRACKING_SETTLE_POLL_S)
            if feedback.shape != (7,) or not np.all(np.isfinite(feedback)):
                progress.close(f"第 {index} 点反馈无效")
                self.hold()
                raise SafetyAbort(
                    "轨迹执行关节反馈含非有限数或维度无效，已停止继续下发"
                )
            tracking_error = float(np.max(np.abs(feedback - target)))
            if not np.isfinite(tracking_error) or tracking_error > tracking_tolerance_deg:
                progress.close(f"第 {index} 点跟踪超差")
                raise SafetyAbort(
                    "轨迹执行反馈偏差过大，已停止继续下发: "
                    f"点 {index}/{len(execution_points)} 最大关节差="
                    f"{tracking_error:.2f}°，上限={tracking_tolerance_deg:.2f}°"
                )
        progress.close()

    def gripper_state(self, timeout_s: float = 3.0) -> dict:
        """Read the installed RM Plus end-effector, not the legacy gripper API."""
        deadline = time.monotonic() + timeout_s
        while True:
            rc, state = self.arm.rm_get_rm_plus_state_info()
            if rc == 0:
                break
            if time.monotonic() >= deadline:
                raise SafetyAbort(f"读取 RM Plus 夹爪状态失败: {rc}")
            time.sleep(0.1)
        if int(state.get("sys_state", 0)) != 0:
            raise SafetyAbort(f"RM Plus 系统状态异常: {state}")
        dof_err = state.get("dof_err", [0])
        if dof_err and int(dof_err[0]) != 0:
            raise SafetyAbort(f"RM Plus 夹爪故障: {state}")
        return state

    def _command_gripper_position(
        self,
        target: int,
        *,
        speed: int,
        force: int | None,
        timeout_s: float = 5.0,
    ) -> dict:
        """Command RM Plus and wait for a fresh, settled state sample."""
        before = self.gripper_state()
        start_pos = int(before["pos"][0])
        if force is not None and self.arm.rm_set_hand_force(int(force)) != 0:
            raise SafetyAbort("设置 RM Plus 夹爪内部力限幅失败")
        if self.arm.rm_set_hand_speed(int(speed)) != 0:
            raise SafetyAbort("设置 RM Plus 夹爪速度失败")
        command = [int(target), -1, -1, -1, -1, -1]
        if self.arm.rm_set_hand_follow_pos(command, False) != 0:
            raise SafetyAbort("RM Plus 夹爪位置命令发送失败")

        deadline = time.monotonic() + timeout_s
        movement_seen = abs(start_pos - target) <= 8
        settled_samples = 0
        stopped_fault_samples = 0
        moving_fault_seen = False
        latest = before
        while time.monotonic() < deadline:
            if self.stop_event.is_set():
                raise SafetyAbort(stop_reason(self.stop_event))
            latest = self.gripper_state()
            state = int(latest["dof_state"][0])
            pos = int(latest["pos"][0])
            speed_now = int(latest["speed"][0])
            if abs(pos - start_pos) >= 8 or state == 0:
                movement_seen = True
            if state in (5, 6):
                # RM Plus may publish an internally inconsistent transition
                # sample while reversing direction: the 2026-07-17 robot run
                # reported state=6 with sys_state=0, dof_err=0 and speed=74,
                # then aborted halfway through reopening.  A real protection
                # stop/fault must remain stopped; do not classify one moving
                # sample as terminal.  sys_state/dof_err are still rejected
                # immediately by gripper_state().
                settled_samples = 0
                if speed_now == 0:
                    stopped_fault_samples += 1
                    if stopped_fault_samples >= 3:
                        raise SafetyAbort(
                            "RM Plus 夹爪连续 3 帧处于保护或故障且已停住: "
                            f"{latest}"
                        )
                else:
                    if not moving_fault_seen:
                        LOG.warning(
                            "RM Plus 夹爪运动中出现瞬态 state=%d pos=%d "
                            "speed=%d（sys_state/dof_err 正常），继续等待终态",
                            state,
                            pos,
                            speed_now,
                        )
                    moving_fault_seen = True
                    stopped_fault_samples = 0
                time.sleep(0.05)
                continue
            stopped_fault_samples = 0
            if moving_fault_seen:
                LOG.info(
                    "RM Plus 夹爪状态恢复: state=%d pos=%d speed=%d",
                    state,
                    pos,
                    speed_now,
                )
                moving_fault_seen = False
            if movement_seen and state in (2, 3) and speed_now == 0:
                settled_samples += 1
                if settled_samples >= 3:
                    return latest
            else:
                settled_samples = 0
            time.sleep(0.05)
        raise SafetyAbort(
            "RM Plus 夹爪动作超时: "
            f"target={target}, pos={latest.get('pos')}, "
            f"state={latest.get('dof_state')}"
        )

    def calibrate_empty_close(self, params: DemoParams | None = None) -> int:
        """在自由空间实测今天的空夹闭合位置，作为后续抓取判定的基线。

        写死的空夹常量会随环境/夹爪状态漂移（2026-07-15 实测因此把真实
        抓取成功误判成空夹）。必须在夹爪前方无物体时调用（例如观察位）。
        结束时夹爪保持张开。
        """
        if not self.take_control:
            raise SafetyAbort("只规划会话禁止控制夹爪")
        params = params or DemoParams()
        opened = self.open_gripper(params)
        opened_pos = int(opened["pos"][0])
        state = self._command_gripper_position(
            params.gripper_close_position,
            speed=params.gripper_speed,
            force=params.gripper_force,
        )
        baseline = int(state["pos"][0])
        travel = opened_pos - baseline
        if travel < params.gripper_calibration_min_travel:
            self.open_gripper(params)
            raise SafetyAbort(
                "空夹基线标定异常，闭合行程不足"
                "（夹爪前方可能有物体或硬件故障）: "
                f"张开 pos={opened_pos}, 闭合 pos={baseline}, "
                f"行程={travel}, 最小要求={params.gripper_calibration_min_travel}"
            )
        self.empty_close_pos = baseline
        LOG.info(
            "空夹基线实测 pos=%d（张开 pos=%d，闭合行程=%d），"
            "抓取判定阈值 pos>%d",
            baseline,
            opened_pos,
            travel,
            baseline + params.gripper_object_margin,
        )
        self.open_gripper(params)
        return baseline

    def open_gripper(self, params: DemoParams | None = None) -> dict:
        if not self.take_control:
            raise SafetyAbort("只规划会话禁止控制夹爪")
        params = params or DemoParams()
        state = self._command_gripper_position(
            params.gripper_open_position,
            speed=params.gripper_speed,
            force=None,
        )
        pos = int(state["pos"][0])
        dof_state = int(state["dof_state"][0])
        validate_open_gripper_feedback(state, params)
        LOG.info("RM Plus 夹爪已打开: state=%d pos=%d", dof_state, pos)
        return state

    def close_gripper(self, params: DemoParams | None = None) -> dict:
        if not self.take_control:
            raise SafetyAbort("只规划会话禁止控制夹爪")
        params = params or DemoParams()
        state = self._command_gripper_position(
            params.gripper_close_position,
            speed=params.gripper_speed,
            force=params.gripper_force,
        )
        dof_state = int(state["dof_state"][0])
        pos = int(state["pos"][0])
        current = int(state["current"][0])
        baseline = getattr(
            self, "empty_close_pos", params.gripper_empty_closed_position
        )
        minimum_object_pos = baseline + params.gripper_object_margin
        LOG.info(
            "RM Plus 闭合反馈: state=%d pos=%d current=%d; "
            "空夹基线=%d(%s) 抓取阈值 pos>%d",
            dof_state,
            pos,
            current,
            baseline,
            "本轮实测" if hasattr(self, "empty_close_pos") else "静态回退",
            minimum_object_pos,
        )
        validate_holding_gripper_feedback(
            state,
            params,
            empty_close_pos=baseline,
        )
        return state

    def close_empty_gripper(self, params: DemoParams | None = None) -> dict:
        """收拢已释放物体的空夹爪，不执行“是否抓到物体”的判定。

        `close_gripper()` 是抓取动作，空夹闭合会被它有意判成抓取失败；放瓶并
        退开后的收纳动作语义不同，只需等待 RM Plus 返回稳定终态。
        """
        if not self.take_control:
            raise SafetyAbort("只规划会话禁止控制夹爪")
        params = params or DemoParams()
        state = self._command_gripper_position(
            params.gripper_close_position,
            speed=params.gripper_speed,
            force=params.gripper_force,
        )
        dof_state = int(state["dof_state"][0])
        pos = int(state["pos"][0])
        LOG.info("RM Plus 空载收拢完成: state=%d pos=%d", dof_state, pos)
        return state

    def hold(self):
        if not self.take_control:
            return
        try:
            self.arm.rm_set_arm_slow_stop()
        except Exception:
            pass

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.arm.rm_delete_robot_arm()
        finally:
            if self.take_control:
                LOG.warning(
                    "SDK 已断开；未自动恢复遥操。检查现场后手动运行官方 upstart_all.sh"
                )
