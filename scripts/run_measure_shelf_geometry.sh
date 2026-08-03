#!/usr/bin/env bash
set -euo pipefail

# 现场货架测量（零机械臂运动，只起头部相机）。
#
# 用法： scripts/run_measure_shelf_geometry.sh <X> <Y> <Z> [faces]
#   X Y Z   目标大致坐标（right_controller_base 系，米），比如把一个瓶子/
#           标记物放在要测的格位，从之前的定位日志里读出的大致位置
#   faces   逗号分隔的货架面列表，默认全部五个
#           (shelf_bottom,shelf_top,shelf_back,shelf_left_panel,shelf_right_panel)
#
# 产出草稿 keepout_boxes 供人工核对，不会自动写 safety_profiles.json。

ROBOT_HOST="${ROBOT_HOST:-rm@192.168.3.68}"
REMOTE_DIR="${REMOTE_DIR:-/home/rm/Grabber}"
REMOTE_PY="${REMOTE_PY:-/home/rm/miniconda3/envs/tube_vision/bin/python}"
FACES="${4:-shelf_bottom,shelf_top,shelf_back,shelf_left_panel,shelf_right_panel}"
MIN_GAP_M="${MIN_GAP_M:-}"
MAX_GAP_M="${MAX_GAP_M:-}"

usage() {
  echo "用法: $0 <X> <Y> <Z> [faces]"
}

fail_config() {
  echo "启动参数无效: $1" >&2
  exit 1
}

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  usage >&2
  exit 1
fi

for value in "$1" "$2" "$3"; do
  if [[ ! "${value}" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
    fail_config "目标坐标必须是数字: ${value}"
  fi
done
if [[ ! "${ROBOT_HOST}" =~ ^([A-Za-z0-9][A-Za-z0-9._-]*@)?[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  fail_config "ROBOT_HOST 只允许 SSH 用户、主机名或 IPv4 地址"
fi
if [[ ! "${REMOTE_DIR}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
  fail_config "REMOTE_DIR 必须是不含空格/引号的绝对路径"
fi
if [[ ! "${REMOTE_PY}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
  fail_config "REMOTE_PY 必须是不含空格/引号的绝对路径"
fi
if [[ ! "${FACES}" =~ ^[A-Za-z0-9_,]+$ ]]; then
  fail_config "faces 只允许字母、数字、下划线和逗号"
fi
if [[ -n "${MIN_GAP_M}" && ! "${MIN_GAP_M}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  fail_config "MIN_GAP_M 必须是非负数"
fi
if [[ -n "${MAX_GAP_M}" && ! "${MAX_GAP_M}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  fail_config "MAX_GAP_M 必须是非负数"
fi

TARGET_X="$1"
TARGET_Y="$2"
TARGET_Z="$3"

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
    shelf_dispenser/ \
    scripts/measure_shelf_geometry.py \
    sensors/camera_thread.py \
    "${ROBOT_HOST}:${REMOTE_DIR}/"
)

EXTRA_ARGS=""
if [[ -n "${MIN_GAP_M}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --min-gap-m ${MIN_GAP_M}"
fi
if [[ -n "${MAX_GAP_M}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --max-gap-m ${MAX_GAP_M}"
fi

echo "== 现场测量货架面（零机械臂运动） =="
# shellcheck disable=SC2029
ssh "${SSH_OPTIONS[@]}" -tt "${ROBOT_HOST}" \
  "cd '${REMOTE_DIR}' && \
   '${REMOTE_PY}' scripts/measure_shelf_geometry.py \
     --target-base '${TARGET_X}' '${TARGET_Y}' '${TARGET_Z}' \
     --faces '${FACES}'${EXTRA_ARGS}"
