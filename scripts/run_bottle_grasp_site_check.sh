#!/usr/bin/env bash
set -euo pipefail

# 板房实机体检（零机械臂运动）。上真机任务前的标准动作：
# 时钟 → 头部舵机基准 → MoveIt 碰撞三态探针 → 相机/SDK/MoveIt 栈 →
# 右臂健康 → 夹爪反馈 → 头部深度流 → 真实 YOLO 定位 → 桌面拟合+场景预算 →
# 完整规划彩排（真实场景+真实 IK+全部契约，产出可执行轨迹）→ 右腕相机流。
# 全绿输出 GO（退出码 0），任何一项失败输出 NO-GO（退出码 2）。
#
# 用法： scripts/run_bottle_grasp_site_check.sh
# 通过后再跑： scripts/run_bottle_grasp.sh {from-observation|from-start}

ROBOT_HOST="${ROBOT_HOST:-rm@192.168.3.68}"
REMOTE_DIR="${REMOTE_DIR:-/home/rm/Grabber}"
REMOTE_PY="${REMOTE_PY:-/home/rm/miniconda3/envs/tube_vision/bin/python}"
SAFETY_PROFILE="${SAFETY_PROFILE:-table_demo}"
PORT="${PORT:-8879}"

fail_config() {
  echo "启动参数无效: $1" >&2
  exit 1
}

if [[ ! "${ROBOT_HOST}" =~ ^([A-Za-z0-9][A-Za-z0-9._-]*@)?[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  fail_config "ROBOT_HOST 只允许 SSH 用户、主机名或 IPv4 地址"
fi
if [[ ! "${REMOTE_DIR}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
  fail_config "REMOTE_DIR 必须是不含空格/引号的绝对路径"
fi
if [[ ! "${REMOTE_PY}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
  fail_config "REMOTE_PY 必须是不含空格/引号的绝对路径"
fi
if [[ ! "${SAFETY_PROFILE}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  fail_config "SAFETY_PROFILE 只允许字母、数字、点、下划线和连字符"
fi
if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || ((10#${PORT} < 1 || 10#${PORT} > 65535)); then
  fail_config "PORT 必须是 1-65535 的整数"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_OPTIONS=(
  -o ConnectTimeout=8
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
)
RSYNC_SSH="ssh -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"

echo "== 同步当前代码到机器人（不删除远端文件） =="
(
  cd "${SCRIPT_DIR}/.."
  rsync -azR --timeout=30 -e "${RSYNC_SSH}" \
    --exclude='__pycache__/' --exclude='*.pyc' \
    bottle_grasp/ \
    scripts/bottle_grasp_demo.py \
    scripts/bottle_grasp_site_check.py \
    sensors/camera_thread.py \
    "${ROBOT_HOST}:${REMOTE_DIR}/"
)

echo "== 相机占用预检（头部 + 右腕） =="
# shellcheck disable=SC2029
ssh "${SSH_OPTIONS[@]}" -tt "${ROBOT_HOST}" \
  "cd '${REMOTE_DIR}' && \
   '${REMOTE_PY}' -m bottle_grasp.camera_access \
     --config config.yaml --camera head --camera right_wrist --no-probe"

echo "== 板房体检（零机械臂运动） =="
# shellcheck disable=SC2029
ssh "${SSH_OPTIONS[@]}" -tt "${ROBOT_HOST}" \
  "cd '${REMOTE_DIR}' && \
   '${REMOTE_PY}' scripts/bottle_grasp_site_check.py \
     --safety-profile '${SAFETY_PROFILE}' --port '${PORT}'"
