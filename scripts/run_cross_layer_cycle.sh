#!/bin/bash
# 跨层完整循环：归位 -> 抓取 -> 持瓶回位 -> 升降 647->250 -> 空位 -> 放置
#
# 这个脚本以前只活在机器人的 /home/rm/cycle.sh，不在任何版本控制里。同步
# 工具查不到它，所以代码更新了而它没更新时，症状是"跑了没反应"，查起来
# 要绕一大圈。现在它跟着仓库走。
#
# 每个阶段的输入都在消费前就地重采，把新鲜度窗口压到最短。抓取和放置各自
# 重试，因为它们依赖感知和随机采样规划；归位和升降不重试，那两步是确定的，
# 失败就是真失败。
#
# 用法（在机器人上）：
#   bash scripts/run_cross_layer_cycle.sh
#
# 跑之前先在 Mac 上确认机器人跑的是同一份代码：
#   python scripts/robot_code_drift.py --push

set -u

G=${SHELF_ROOT:-/home/rm/dual-arm-shelf-dispenser}
PY=${SHELF_PYTHON:-/home/rm/miniconda3/envs/tube_vision/bin/python3}
O=${CYCLE_OUT:-/home/rm/cycle}
SPEED=${CYCLE_SPEED:-100}

# $O gets wiped, so only ever wipe a directory this script created.  Comparing
# against $HOME or the repo path is not enough -- it depends on who is running
# where, and CYCLE_OUT=/home/rm would have deleted the robot's home.
MARKER="$O/.cycle_output"
[ "${O#/}" = "$O" ] && { echo "拒绝: CYCLE_OUT 必须是绝对路径" >&2; exit 1; }
if [ -e "$O" ] && [ ! -f "$MARKER" ]; then
  echo "拒绝: $O 已存在且不是本脚本建的输出目录（缺 .cycle_output 标记）" >&2
  echo "      要复用它，先自己清空；要换目录，设 CYCLE_OUT。" >&2
  exit 1
fi
rm -rf "$O"; mkdir -p "$O"; touch "$MARKER"
say(){ echo; echo "########## $* ##########"; }
fail(){ echo "  失败: $(grep -E '拒绝|SafetyAbort' "$1" | tail -1)"; tail -4 "$1"; echo CYCLE_DONE; exit 1; }

say "阶段 1  归位到抓取起点"
cd "$G"
if $PY scripts/normalize_to_grasp_start.py --execute > "$O/norm.log" 2>&1; then
  echo "  OK"
else
  fail "$O/norm.log"
fi

say "阶段 2  抓取（标定+采集+规划+执行 都在同一次尝试内）"
PICKED=0
for i in 1 2 3 4 5 6; do
  echo "  --- 尝试 $i ---"
  cd "$G"
  $PY scripts/calibrate_mtc_gripper.py --record "$O/grip.json" --execute > "$O/cal$i.log" 2>&1 \
    || { echo "    夹爪标定失败: $(grep -E '拒绝|SafetyAbort' "$O/cal$i.log" | tail -1)"; continue; }
  $PY scripts/capture_mtc_direct_pick_scene.py --scenario-out "$O/p$i.yaml" > "$O/pc$i.log" 2>&1 \
    || { echo "    采集失败: $(grep -E '拒绝|Error' "$O/pc$i.log" | tail -1)"; continue; }
  cd "$G/mtc_ws" && source install/setup.bash
  rm -f "$O/p$i.json"*
  ros2 launch grabber_mtc_planner plan_shelf_transfer_experimental.launch.py \
    scenario:="$O/p$i.yaml" out:="$O/p$i.json" hold_seconds:=0 > "$O/pp$i.log" 2>&1 || true
  [ -f "$O/p$i.json" ] || { echo "    规划无结果"; continue; }
  if [ "$($PY -c "import json;print(json.load(open('$O/p$i.json')).get('solved'))")" != "True" ]; then
    echo "    未解出: $($PY -c "import json;d=json.load(open('$O/p$i.json'));print(list(d.get('earliest_failure_stage_by_arm',{}).values())[:1])")"
    continue
  fi
  echo "    规划成功，执行抓取"
  cd "$G"
  if $PY scripts/execute_mtc_trajectory.py pick --result "$O/p$i.json" \
      --trajectory "$O/p$i.json.trajectory.json" --scenario "$O/p$i.yaml" \
      --gripper-calibration-record "$O/grip.json" --record "$O/pick_record.json" \
      --speed "$SPEED" --allow-sdk-retiming --execute > "$O/px$i.log" 2>&1; then
    echo "    抓取成功"; PICKED=1; break
  else
    echo "    执行失败: $(grep -E '拒绝|SafetyAbort' "$O/px$i.log" | tail -1)"
  fi
done
[ "$PICKED" = 1 ] || { echo "抓取阶段失败"; echo CYCLE_DONE; exit 1; }

say "阶段 2.5  持瓶回收拢位（升降契约要求的姿态，并把到位证据写回 pick 记录）"
cd "$G"
if $PY scripts/normalize_to_grasp_start.py --target carry_home \
    --pick-record "$O/pick_record.json" --execute > "$O/tuck.log" 2>&1; then
  echo "  OK"
else
  fail "$O/tuck.log"
fi

say "阶段 3  升降 647 -> 250（持瓶）——首次上硬件，人守在急停旁"
cd "$G"
if $PY scripts/execute_mtc_lift_transfer.py "$O/pick_record.json" \
    --record "$O/lift_record.json" --speed 30 --execute > "$O/lift.log" 2>&1; then
  echo "  OK"
else
  fail "$O/lift.log"
fi

say "阶段 4  放置（空位+场景+规划+执行 都在同一次尝试内）"
for i in 1 2 3 4 5; do
  echo "  --- 尝试 $i ---"
  cd "$G"
  $PY scripts/capture_empty_shelf_places.py --expected-lift-mm 250 \
    --roi-min -0.40 0.58 -0.45 --roi-max 0.23 0.72 0.05 \
    --lift-execution-record "$O/lift_record.json" \
    --output "$O/pl$i.json" > "$O/plc$i.log" 2>&1 \
    || { echo "    空位采集失败: $(grep -E '拒绝|Error' "$O/plc$i.log" | tail -1)"; continue; }
  $PY scripts/empty_shelf_places_to_mtc_scenario.py "$O/pl$i.json" "$O/pls$i.yaml" > "$O/plg$i.log" 2>&1 \
    || { echo "    场景生成失败: $(tail -1 "$O/plg$i.log")"; continue; }
  cd "$G/mtc_ws" && source install/setup.bash
  rm -f "$O/pls${i}_res.json"*
  ros2 launch grabber_mtc_planner plan_shelf_transfer_experimental.launch.py \
    scenario:="$O/pls$i.yaml" out:="$O/pls${i}_res.json" hold_seconds:=0 > "$O/plp$i.log" 2>&1 || true
  [ -f "$O/pls${i}_res.json" ] || { echo "    规划无结果"; continue; }
  if [ "$($PY -c "import json;print(json.load(open('$O/pls${i}_res.json')).get('solved'))")" != "True" ]; then
    echo "    未解出: $($PY -c "import json;d=json.load(open('$O/pls${i}_res.json'));print(list(d.get('earliest_failure_stage_by_arm',{}).values())[:1])")"
    continue
  fi
  echo "    规划成功，执行放置"
  cd "$G"
  if $PY scripts/execute_mtc_trajectory.py place --result "$O/pls${i}_res.json" \
      --trajectory "$O/pls${i}_res.json.trajectory.json" --scenario "$O/pls$i.yaml" \
      --record "$O/place_record.json" --speed "$SPEED" --allow-sdk-retiming --execute > "$O/plx$i.log" 2>&1; then
    echo "    放置成功"; echo CYCLE_SUCCESS; exit 0
  else
    echo "    执行失败: $(grep -E '拒绝|SafetyAbort' "$O/plx$i.log" | tail -1)"
  fi
done
echo "放置阶段失败"
echo CYCLE_DONE
exit 1
