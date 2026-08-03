#!/usr/bin/env bash
set -euo pipefail

# The only public real-robot bottle task launcher.
#
#   from-pregrasp:   right arm is already at the verified 8.5 cm hover after
#                     a safe abort; fresh head+wrist lock -> final grasp tail
#   from-observation: right arm is already at a usable wrist observation pose;
#                     fresh head+wrist lock -> grasp -> lift -> place -> retreat
#   from-start:       fresh head lock -> MoveIt transfer to observation pose ->
#                     the same grasp/place tail -> return to configured home

ROBOT_HOST="${ROBOT_HOST:-rm@192.168.3.68}"
REMOTE_DIR="${REMOTE_DIR:-/home/rm/Grabber}"
REMOTE_PY="${REMOTE_PY:-/home/rm/miniconda3/envs/tube_vision/bin/python}"
SAFETY_PROFILE="${SAFETY_PROFILE:-table_demo}"
PORT="${PORT:-8879}"
DISPENSE="${DISPENSE:-0}"
DELIVERY_SAFETY_PROFILE="${DELIVERY_SAFETY_PROFILE:-}"
TARGET_PRODUCT="${TARGET_PRODUCT:-}"
VISUAL_SERVO="${VISUAL_SERVO:-0}"

# Do not collapse an explicitly supplied empty value into a default for the
# new safety controls.  An unset value means "use the established default";
# an empty or malformed value is a configuration error that must be caught
# before this launcher can reach rsync/SSH.
if [[ "${COMMISSIONING_SPEED+x}" == "x" ]]; then
  EFFECTIVE_COMMISSIONING_SPEED="${COMMISSIONING_SPEED}"
else
  EFFECTIVE_COMMISSIONING_SPEED=""
fi
if [[ "${BOTTLE_GRASP_TRAJECTORY_MODE+x}" == "x" ]]; then
  EFFECTIVE_TRAJECTORY_MODE="${BOTTLE_GRASP_TRAJECTORY_MODE}"
else
  EFFECTIVE_TRAJECTORY_MODE="continuous"
fi
if [[ "${VISUAL_MODE+x}" == "x" ]]; then
  EFFECTIVE_VISUAL_MODE="${VISUAL_MODE}"
else
  # Keep the original public switch working for existing runbook commands.
  case "${VISUAL_SERVO}" in
    0) EFFECTIVE_VISUAL_MODE="off" ;;
    1) EFFECTIVE_VISUAL_MODE="active" ;;
    *) EFFECTIVE_VISUAL_MODE="" ;;
  esac
fi
if [[ "${STOP_AFTER_OBSERVATION+x}" == "x" ]]; then
  EFFECTIVE_STOP_AFTER_OBSERVATION="${STOP_AFTER_OBSERVATION}"
else
  EFFECTIVE_STOP_AFTER_OBSERVATION="0"
fi
if [[ "${CONFIRM_BEFORE_GRASP+x}" == "x" ]]; then
  EFFECTIVE_CONFIRM_BEFORE_GRASP="${CONFIRM_BEFORE_GRASP}"
else
  EFFECTIVE_CONFIRM_BEFORE_GRASP="0"
fi

usage() {
  echo "用法: $0 {from-pregrasp|from-observation|from-start}"
  echo "  from-pregrasp    右臂已在 8.5cm 预抓取悬停位，复核后续跑最后抓放段"
  echo "  from-observation  右臂已经在观察位，从本轮新鲜定位开始完整抓放"
  echo "  from-start        从固定头部定位开始完整抓放并返回初始姿态"
  echo
  echo "可选环境变量（均在同步/SSH 前严格校验）："
  echo "  COMMISSIONING_SPEED=1..100             为有人监管的调试运行限制所有运动速度"
  echo "  BOTTLE_GRASP_TRAJECTORY_MODE=continuous|blocking"
  echo "                                          continuous 为默认；blocking 是 SDK 回退"
  echo "  VISUAL_MODE=off|shadow|active           默认由旧 VISUAL_SERVO=0|1 推导"
  echo "  VISUAL_SERVO=0|1                        兼容旧命令（0=off，1=active）"
  echo "  STOP_AFTER_OBSERVATION=0|1              到观察位后结束（不闭夹）"
  echo "  CONFIRM_BEFORE_GRASP=0|1                定位后在终端确认再抓取"
}

fail_config() {
  echo "启动参数无效: $1" >&2
  exit 1
}

capture_local_source_provenance() {
  # The robot receives source files through rsync, not the local .git
  # directory.  Freeze provenance here, on the source checkout, so the
  # immutable remote evidence bundle identifies exactly what was sent.
  local provenance
  if ! provenance="$(
    cd "${PROJECT_ROOT}"
    python3 -c '
import runpy

# Loading the module by path avoids bottle_grasp/__init__.py (and therefore
# avoids requiring the full vision/robot Python environment on the Mac just
# to hash local Git evidence).
record = runpy.run_path("bottle_grasp/run_manifest.py")[
    "collect_git_provenance"
](".")
if record.get("state") != "available":
    raise SystemExit("local git provenance is unavailable")
values = (
    str(record.get("commit_sha") or ""),
    "1" if record.get("dirty") else "0",
    str(record.get("dirty_digest") or ""),
    str(record.get("dirty_digest_algorithm") or ""),
)
if not all(values):
    raise SystemExit("local git provenance is incomplete")
print("\t".join(values))
'
  )"; then
    fail_config "无法捕获本地 Git 溯源；不会同步或连接机器人"
  fi

  IFS=$'\t' read -r \
    SOURCE_GIT_SHA SOURCE_DIRTY SOURCE_DIRTY_DIGEST \
    SOURCE_DIRTY_DIGEST_ALGORITHM <<< "${provenance}" \
    || fail_config "本地 Git 溯源输出无效；不会同步或连接机器人"

  if [[ ! "${SOURCE_GIT_SHA}" =~ ^[0-9a-fA-F]+$ ]] || (( ${#SOURCE_GIT_SHA} != 40 && ${#SOURCE_GIT_SHA} != 64 )); then
    fail_config "本地 Git SHA 无效；不会同步或连接机器人"
  fi
  if [[ ! "${SOURCE_DIRTY}" =~ ^[01]$ ]]; then
    fail_config "本地 Git dirty 标记无效；不会同步或连接机器人"
  fi
  if [[ ! "${SOURCE_DIRTY_DIGEST}" =~ ^[0-9a-fA-F]{64}$ ]]; then
    fail_config "本地 Git dirty digest 无效；不会同步或连接机器人"
  fi
  if [[ ! "${SOURCE_DIRTY_DIGEST_ALGORITHM}" =~ ^[A-Za-z0-9._+-]+$ ]]; then
    fail_config "本地 Git dirty digest 算法无效；不会同步或连接机器人"
  fi
}

if [[ "$#" -ne 1 ]]; then
  usage >&2
  exit 1
fi

# These values are later embedded in rsync/SSH destinations and a remote shell
# command.  Keep the supported override surface deliberately narrow instead of
# accepting whitespace, quotes or shell metacharacters.
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
if [[ ! "${PORT}" =~ ^[0-9]+$ ]]; then
  fail_config "PORT 必须是 1-65535 的整数"
fi
if [[ "${DISPENSE}" != "0" && "${DISPENSE}" != "1" ]]; then
    fail_config "DISPENSE 只能是 0 或 1"
fi
if [[ "${VISUAL_SERVO}" != "0" && "${VISUAL_SERVO}" != "1" ]]; then
  fail_config "VISUAL_SERVO 只能是 0 或 1"
fi
if [[ "${COMMISSIONING_SPEED+x}" == "x" && ! "${EFFECTIVE_COMMISSIONING_SPEED}" =~ ^([1-9][0-9]?|100)$ ]]; then
  fail_config "COMMISSIONING_SPEED 必须是 1-100 的整数；不设置则保留默认速度"
fi
case "${EFFECTIVE_TRAJECTORY_MODE}" in
  continuous)
    CONTINUOUS_TRAJECTORY="1"
    ;;
  blocking)
    CONTINUOUS_TRAJECTORY="0"
    ;;
  *)
    fail_config "BOTTLE_GRASP_TRAJECTORY_MODE 只能是 continuous 或 blocking"
    ;;
esac
case "${EFFECTIVE_VISUAL_MODE}" in
  off|shadow|active)
    ;;
  *)
    fail_config "VISUAL_MODE 只能是 off、shadow 或 active"
    ;;
esac
if [[ "${EFFECTIVE_STOP_AFTER_OBSERVATION}" != "0" && "${EFFECTIVE_STOP_AFTER_OBSERVATION}" != "1" ]]; then
  fail_config "STOP_AFTER_OBSERVATION 只能是 0 或 1"
fi
if [[ "${EFFECTIVE_CONFIRM_BEFORE_GRASP}" != "0" && "${EFFECTIVE_CONFIRM_BEFORE_GRASP}" != "1" ]]; then
  fail_config "CONFIRM_BEFORE_GRASP 只能是 0 或 1"
fi
if [[ "${EFFECTIVE_STOP_AFTER_OBSERVATION}" == "1" && "${EFFECTIVE_CONFIRM_BEFORE_GRASP}" == "1" ]]; then
  fail_config "STOP_AFTER_OBSERVATION=1 与 CONFIRM_BEFORE_GRASP=1 互斥"
fi
if [[ -n "${DELIVERY_SAFETY_PROFILE}" && ! "${DELIVERY_SAFETY_PROFILE}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  fail_config "DELIVERY_SAFETY_PROFILE 只允许字母、数字、点、下划线和连字符"
fi
if [[ "${DISPENSE}" == "1" && -z "${DELIVERY_SAFETY_PROFILE}" ]]; then
  fail_config "DISPENSE=1 时必须设置 DELIVERY_SAFETY_PROFILE"
fi
if [[ -n "${TARGET_PRODUCT}" && ! "${TARGET_PRODUCT}" =~ ^[A-Za-z0-9_,-]+$ ]]; then
  fail_config "TARGET_PRODUCT 只允许字母、数字、逗号、下划线和连字符"
fi
if ((10#${PORT} < 1 || 10#${PORT} > 65535)); then
  fail_config "PORT 必须是 1-65535 的整数"
fi

MODE="$1"
case "${MODE}" in
  from-pregrasp)
    MODE_NOTE="当前预抓取悬停位 -> 最后接近 -> 抓取 -> 抬升 -> 放回 -> 退开"
    ;;
  from-observation)
    MODE_NOTE="当前右臂观察位 -> 抓取 -> 抬升 -> 放回 -> 退开"
    ;;
  from-start)
    MODE_NOTE="头部定位 -> 避障到观察位 -> 抓取 -> 抬升 -> 放回 -> 返回初始姿态"
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
if [[ "${MODE}" == "from-pregrasp" && "${EFFECTIVE_STOP_AFTER_OBSERVATION}" == "1" ]]; then
  fail_config "STOP_AFTER_OBSERVATION=1 不支持 from-pregrasp：该入口已越过观察位"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
capture_local_source_provenance

echo
echo "!! 即将执行真机任务: ${MODE_NOTE}"
if [[ -n "${EFFECTIVE_COMMISSIONING_SPEED}" ]]; then
  echo "   commissioning 速度上限: ${EFFECTIVE_COMMISSIONING_SPEED}%（远端 --commissioning-speed）"
else
  echo "   commissioning 速度上限: 未设置（保留现有速度默认值）"
fi
echo "   MoveIt 轨迹执行: ${EFFECTIVE_TRAJECTORY_MODE}（BOTTLE_GRASP_CONTINUOUS_TRAJECTORY=${CONTINUOUS_TRAJECTORY}）"
case "${EFFECTIVE_VISUAL_MODE}" in
  active)
    echo "   预抓取视觉闭环: active（默认 8mm/步、15mm 累计、最多 2 次）"
    ;;
  shadow)
    echo "   预抓取视觉闭环: shadow（只记录建议修正，不应用腕部修正动作）"
    ;;
  off)
    echo "   预抓取视觉闭环: off（原路径，A/B 基线）"
    ;;
esac
if [[ "${EFFECTIVE_STOP_AFTER_OBSERVATION}" == "1" ]]; then
  echo "   阶段入口: stop-after-observation（到观察位和定位后结束；不闭夹）"
elif [[ "${EFFECTIVE_CONFIRM_BEFORE_GRASP}" == "1" ]]; then
  echo "   阶段入口: confirm-before-grasp（定位后等待终端确认）"
else
  echo "   阶段入口: 完整 task-mode 默认流程"
fi
echo "   本地源码: ${SOURCE_GIT_SHA}；dirty=${SOURCE_DIRTY}；dirty digest=${SOURCE_DIRTY_DIGEST}"
echo "   确认桌面/瓶子布置正确、机械臂周围清空，并且有人手放在硬件急停上。"
echo "   现在开始拍视频。"
echo "   视频已开始且急停就位后，输入：开始"
echo "   其他任何输入（包括只按 Enter）都会取消；Ctrl+C 也可取消。"
read -r VIDEO_CONFIRM
if [[ "${VIDEO_CONFIRM}" != "开始" ]]; then
  echo "未收到录像与急停确认，已取消，机器人不会开始任务。" >&2
  exit 2
fi

SSH_OPTIONS=(
  -o ConnectTimeout=8
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
)
RSYNC_SSH="ssh -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"

echo "== 同步当前 bottle task 代码到机器人（不删除远端文件） =="
(
  cd "${SCRIPT_DIR}/.."
  rsync -azR --timeout=30 -e "${RSYNC_SSH}" \
    --exclude='__pycache__/' --exclude='*.pyc' \
    bottle_grasp/ \
    scripts/bottle_grasp_demo.py \
    scripts/woosh_rotate_relative.cpp \
    scripts/run_bottle_grasp_autonomous.sh \
    scripts/run_bottle_grasp_resume.sh \
    scripts/start_bottle_demo.sh \
    sensors/camera_thread.py \
    "${ROBOT_HOST}:${REMOTE_DIR}/"
)

echo
echo "== 相机占用预检（头部 + 右腕） =="
# shellcheck disable=SC2029
ssh "${SSH_OPTIONS[@]}" -tt "${ROBOT_HOST}" \
  "cd '${REMOTE_DIR}' && \
   '${REMOTE_PY}' -m bottle_grasp.camera_access \
     --config config.yaml --camera head --camera right_wrist --no-probe"

EXTRA_ARGS=""
if [[ -n "${EFFECTIVE_COMMISSIONING_SPEED}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --commissioning-speed '${EFFECTIVE_COMMISSIONING_SPEED}'"
fi
if [[ "${DISPENSE}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --dispense --delivery-safety-profile '${DELIVERY_SAFETY_PROFILE}'"
fi
if [[ -n "${TARGET_PRODUCT}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --target-product '${TARGET_PRODUCT}'"
fi
EXTRA_ARGS="${EXTRA_ARGS} --visual-servo-mode '${EFFECTIVE_VISUAL_MODE}'"
if [[ "${EFFECTIVE_STOP_AFTER_OBSERVATION}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --stop-after-observation"
fi
if [[ "${EFFECTIVE_CONFIRM_BEFORE_GRASP}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --confirm-before-grasp"
fi

echo "== 运行完整任务: ${MODE} =="
# Python already writes latest.log plus the immutable per-run evidence bundle;
# do not tee into latest.log a second time from the shell.
# shellcheck disable=SC2029
ssh "${SSH_OPTIONS[@]}" -tt "${ROBOT_HOST}" \
  "cd '${REMOTE_DIR}' && \
   BOTTLE_GRASP_SOURCE_GIT_SHA='${SOURCE_GIT_SHA}' \
   BOTTLE_GRASP_SOURCE_DIRTY='${SOURCE_DIRTY}' \
   BOTTLE_GRASP_SOURCE_DIRTY_DIGEST='${SOURCE_DIRTY_DIGEST}' \
   BOTTLE_GRASP_SOURCE_DIRTY_DIGEST_ALGORITHM='${SOURCE_DIRTY_DIGEST_ALGORITHM}' \
   BOTTLE_GRASP_CONTINUOUS_TRAJECTORY='${CONTINUOUS_TRAJECTORY}' '${REMOTE_PY}' scripts/bottle_grasp_demo.py \
     --execute --task-mode '${MODE}' \
     --safety-profile '${SAFETY_PROFILE}' --port '${PORT}'${EXTRA_ARGS}"
