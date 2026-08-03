#!/bin/bash
# 首次部署：把这个仓库整份 rsync 到机器人，不需要机器人那边有 GitHub 认证。
#
# 之后的日常同步用 scripts/robot_code_drift.py --push，它只动改过的关键文件、
# 而且会复核哈希。这个脚本是给"机器人上还什么都没有"那一次用的。
#
#   bash scripts/deploy_to_robot.sh              # 先看要传什么，不真传
#   bash scripts/deploy_to_robot.sh --go         # 真传
#
# 机器人上跑起来还需要的两样东西不在这里：ROS 2 工作区要重新 colcon build
# （mtc_ws 的 build/install 不传，那是编译产物，架构也不同），以及 conda 环境
# 已经在机器人上装好了，脚本只检查不安装。

set -euo pipefail

HOST=${ROBOT_HOST:-rm@192.168.3.68}
DEST=${ROBOT_DEST:-/home/rm/dual-arm-shelf-dispenser}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${ROBOT_PYTHON:-/home/rm/miniconda3/envs/tube_vision/bin/python3}

GO=0
[ "${1:-}" = "--go" ] && GO=1

# 传源码和标定/示教数据；不传编译产物、缓存、运行输出、本机虚拟环境。
# .mujoco_assets 是 38M 的机器人描述文件，机器人上本来就有一份，不重复传。
EXCLUDES=(
  --exclude '.git'
  --exclude '__pycache__'
  --exclude '*.pyc'
  --exclude '.pytest_cache'
  --exclude '.mujoco_assets'
  --exclude 'mtc_ws/build'
  --exclude 'mtc_ws/install'
  --exclude 'mtc_ws/log'
  --exclude 'outputs'
  --exclude 'tmp'
)

echo "本地  $ROOT"
echo "机器人 $HOST:$DEST"
echo

if ! ssh -o ConnectTimeout=8 "$HOST" true 2>/dev/null; then
  echo "连不上 $HOST。检查网络/电源，或用 ROBOT_HOST 换地址。" >&2
  exit 1
fi

if [ "$GO" = 0 ]; then
  echo "=== 干跑：以下文件会被传输（加 --go 才真传）==="
  rsync -avn --delete "${EXCLUDES[@]}" "$ROOT/" "$HOST:$DEST/" | tail -40
  echo
  echo "加 --go 执行。注意 --delete：机器人上 $DEST 里不属于本仓库的文件会被删除。"
  exit 0
fi

echo "=== 传输中 ==="
rsync -a --delete --info=stats1 "${EXCLUDES[@]}" "$ROOT/" "$HOST:$DEST/"

echo
echo "=== 机器人上复核 ==="
ssh "$HOST" "
  set -e
  cd '$DEST'
  echo -n '  Python: '; $PY -c 'import sys;print(sys.version.split()[0])'
  echo -n '  一方模块可导入: '
  $PY -c \"import sys;sys.path.insert(0,'.');import shelf_dispenser.safety, shelf_dispenser.arm;print('OK')\"
  echo -n '  安全 profile 可加载: '
  $PY -c \"import sys;sys.path.insert(0,'.')
from shelf_dispenser.safety import load_safety_profile
p=load_safety_profile('shelf_dispenser/safety_profiles.json','shelf_template',require_verified=True)
print('OK', p.name, p.grasp_start_lift_height_mm, 'mm')\"
"

echo
echo "=== 还需要手动做的 ==="
echo "  1. 编译 ROS 2 工作区（编译产物没传，架构不同）："
echo "     ssh $HOST 'cd $DEST/mtc_ws && source /opt/ros/humble/setup.bash && colcon build'"
echo "  2. 确认 .mujoco_assets 在机器人上存在（仅离线仿真需要）"
echo
echo "之后的日常同步改用： python scripts/robot_code_drift.py --push"
