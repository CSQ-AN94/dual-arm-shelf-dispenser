#!/usr/bin/env bash
set -euo pipefail

# Separate supervised experiment launcher.  It intentionally does not call or
# modify the normal run_bottle_grasp.sh safety path.

ROBOT_HOST="${ROBOT_HOST:-rm@192.168.3.68}"
REMOTE_DIR="${REMOTE_DIR:-/home/rm/Grabber}"
REMOTE_PY="${REMOTE_PY:-/home/rm/miniconda3/envs/tube_vision/bin/python}"
PORT="${PORT:-8879}"

usage() {
  echo "用法: $0 {to-observation|from-observation|from-start}"
  echo "  to-observation    只真实移动到右腕观察位并验证，不抓取"
  echo "  此入口关闭桌面、RGB-D 和右腕通道环境避障，仅限有人握住急停的低速实验"
}

fail_config() {
  echo "启动参数无效: $1" >&2
  exit 1
}

if [[ "$#" -ne 1 ]]; then
  usage >&2
  exit 1
fi

if [[ ! "${ROBOT_HOST}" =~ ^([A-Za-z0-9][A-Za-z0-9._-]*@)?[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  fail_config "ROBOT_HOST 只允许 SSH 用户、主机名或 IPv4 地址"
fi
if [[ ! "${REMOTE_DIR}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
  fail_config "REMOTE_DIR 必须是不含空格/引号的绝对路径"
fi
if [[ ! "${REMOTE_PY}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
  fail_config "REMOTE_PY 必须是不含空格/引号的绝对路径"
fi
if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || ((10#${PORT} < 1 || 10#${PORT} > 65535)); then
  fail_config "PORT 必须是 1-65535 的整数"
fi

MODE="$1"
TASK_MODE="${MODE}"
STOP_AFTER_OBSERVATION_FLAG=""
case "${MODE}" in
  to-observation)
    TASK_MODE="from-start"
    STOP_AFTER_OBSERVATION_FLAG="--stop-after-observation"
    MODE_NOTE="头部定位 -> 抬高展开准备位 -> 重新定位 -> 空环境规划并真实移动到观察位 -> 右腕验证 -> 停止（不抓取）"
    ;;
  from-observation)
    MODE_NOTE="当前右臂观察位 -> 抓取 -> 抬升 -> 放回 -> 退开"
    ;;
  from-start)
    MODE_NOTE="头部定位 -> 抬高展开准备位 -> 重新定位 -> 空环境规划到观察位 -> 抓取 -> 抬升 -> 放回 -> 返回初始姿态"
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_OPTIONS=(
  -o ConnectTimeout=8
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
)
RSYNC_SSH="ssh -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"

echo "!! 危险实验模式：环境碰撞保护已关闭"
echo "   ${MODE_NOTE}"
echo "   不检查桌子、杂物或腕部接近通道；机械臂可能按规划直接撞上环境。"
echo "   仅保留关节限位、自碰撞/左臂碰撞、IK/轨迹校验和控制器停止。"
echo "   前后非接触转移为 15%，最终接近/升降为 3%；现场必须清空并由一人全程握住硬件急停。"
echo
read -r -p "按 Enter 或输入 y 继续，其他输入取消: " ACK
case "${ACK}" in
  ""|y|Y)
    ;;
  *)
    echo "已取消。" >&2
    exit 2
    ;;
esac

echo "== 同步独立实验入口到机器人（不删除远端文件） =="
(
  cd "${SCRIPT_DIR}/.."
  rsync -azR --timeout=30 -e "${RSYNC_SSH}" \
    --exclude='__pycache__/' --exclude='*.pyc' \
    bottle_grasp/ \
    scripts/bottle_grasp_no_environment_avoidance.py \
    sensors/camera_thread.py \
    "${ROBOT_HOST}:${REMOTE_DIR}/"
)

echo "== 相机占用预检（头部 + 右腕） =="
# shellcheck disable=SC2029
ssh "${SSH_OPTIONS[@]}" -tt "${ROBOT_HOST}" \
  "cd '${REMOTE_DIR}' && \
   '${REMOTE_PY}' -m bottle_grasp.camera_access \
     --config config.yaml --camera head --camera right_wrist --no-probe"

echo "== 运行无环境避障抓取: ${MODE} =="
# shellcheck disable=SC2029
ssh "${SSH_OPTIONS[@]}" -tt "${ROBOT_HOST}" \
  "cd '${REMOTE_DIR}' && \
   '${REMOTE_PY}' scripts/bottle_grasp_no_environment_avoidance.py \
     --execute --acknowledge-no-environment-collision-check \
     --task-mode '${TASK_MODE}' ${STOP_AFTER_OBSERVATION_FLAG} --port '${PORT}'"
