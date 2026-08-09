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
#   PRODUCT_CODE=P01 bash scripts/run_cross_layer_cycle.sh
#
# 跑之前先在 Mac 上确认机器人跑的是同一份代码：
#   python scripts/robot_code_drift.py --push

set -u

G=${SHELF_ROOT:-/home/rm/dual-arm-shelf-dispenser}
PY=${SHELF_PYTHON:-/home/rm/miniconda3/envs/tube_vision/bin/python3}
O=${CYCLE_OUT:-/home/rm/cycle}
SPEED=${CYCLE_SPEED:-100}
ROW_TEMPLATES=${ROW_TEMPLATES:-$G/outputs/row_templates.json}
PRODUCT_CODE=${PRODUCT_CODE:-}

case "$PRODUCT_CODE" in
  P01|P02|P03|P04|P05|P06) ;;
  *) echo "拒绝: 必须用 PRODUCT_CODE=P01..P06 指明当前瓶型" >&2; exit 1 ;;
esac

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
# Timestamped, and flushed as it happens: watching this file with `tail -f` is
# how an operator standing at the robot can tell "still planning" from "hung",
# and the per-stage clock is what turns "the retract feels slow" into a number.
CYCLE_STARTED=$(date +%s)
say(){
  echo
  echo "########## [$(date +%H:%M:%S) +$(( $(date +%s) - CYCLE_STARTED ))s] $* ##########"
}
fail(){ echo "  失败: $(grep -E '拒绝|SafetyAbort' "$1" | tail -1)"; tail -4 "$1"; echo CYCLE_DONE; exit 1; }

source_mtc_workspace(){
  # colcon's generated setup reads optional variables such as COLCON_TRACE.
  # Temporarily relax nounset while sourcing it, then restore this script's gate.
  set +u
  source /opt/ros/humble/setup.bash
  source /home/rm/ros2_ws/install/setup.bash
  source "$G/mtc_ws/install/setup.bash"
  SOURCE_STATUS=$?
  set -u
  return "$SOURCE_STATUS"
}

# Own the read-only joint-state + move_group stack this workflow depends on.
# The MTC planner launch below deliberately starts only the planner node.
STACK_PID=""
cleanup_stack(){
  [ -n "$STACK_PID" ] || return 0
  if kill -0 -- "-$STACK_PID" 2>/dev/null; then
    kill -INT -- "-$STACK_PID" 2>/dev/null || true
    for _ in {1..24}; do
      kill -0 -- "-$STACK_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -TERM -- "-$STACK_PID" 2>/dev/null || true
  fi
  wait "$STACK_PID" 2>/dev/null || true
  STACK_PID=""
}
trap cleanup_stack EXIT

[ -f "$ROW_TEMPLATES" ] || {
  echo "拒绝: 找不到行抓取模板 $ROW_TEMPLATES" >&2
  exit 1
}

start_stack(){
  STACK_LABEL=$1
  say "启动只规划 MoveIt 栈（$STACK_LABEL）"
  cd "$G/mtc_ws" && source_mtc_workspace
  BRIDGE_STATUS="$O/bridge_status_$STACK_LABEL.json"
  STACK_LOG="$O/moveit_stack_$STACK_LABEL.log"
  rm -f "$BRIDGE_STATUS"
  setsid ros2 launch grabber_robot_state_bridge live_state_plan_only.launch.py \
    bridge_status_file:="$BRIDGE_STATUS" > "$STACK_LOG" 2>&1 &
  STACK_PID=$!
  STACK_READY=0
  for _ in {1..70}; do
    kill -0 "$STACK_PID" 2>/dev/null || break
    if [ -f "$BRIDGE_STATUS" ] && \
        "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("read_only") is True and d.get("publishing") is True and d.get("lift_motion_ready") is True and int(d.get("published", 0)) >= 3 else 1)' "$BRIDGE_STATUS" 2>/dev/null && \
        ros2 service list 2>/dev/null | grep -qx '/get_planning_scene'; then
      STACK_READY=1
      break
    fi
    sleep 0.5
  done
  if [ "$STACK_READY" != 1 ]; then
    echo "拒绝: 实时双臂/升降 joint-state + move_group 在 35 秒内未就绪" >&2
    tail -20 "$STACK_LOG" >&2
    exit 1
  fi
  echo "  OK"
}

say "阶段 0  右臂和升降预归位（左臂保持当前姿态进入碰撞场）"
cd "$G"
if $PY scripts/normalize_to_grasp_start.py --right-and-lift-only \
    --execute > "$O/norm_right_lift.log" 2>&1; then
  echo "  OK"
else
  fail "$O/norm_right_lift.log"
fi

say "阶段 0.5  左臂在标定升降高度归位"
cd "$G"
if $PY scripts/normalize_left_arm.py --execute > "$O/norm_left.log" 2>&1; then
  echo "  OK"
else
  fail "$O/norm_left.log"
fi

say "阶段 1  双臂和升降原子门禁"
cd "$G"
if $PY scripts/normalize_to_grasp_start.py --execute > "$O/norm.log" 2>&1; then
  echo "  OK"
else
  fail "$O/norm.log"
fi

start_stack pick

say "阶段 2  抓取（标定+采集+规划+执行 都在同一次尝试内）"
PICKED=0
for i in 1 2 3 4 5 6; do
  echo "  --- 尝试 $i ---"
  cd "$G"
  $PY scripts/calibrate_mtc_gripper.py --record "$O/grip.json" --execute > "$O/cal$i.log" 2>&1 \
    || { echo "    夹爪标定失败: $(grep -E '拒绝|SafetyAbort' "$O/cal$i.log" | tail -1)"; continue; }
  $PY scripts/capture_mtc_direct_pick_scene.py --target-product "$PRODUCT_CODE" \
    --scenario-out "$O/p$i.yaml" > "$O/pc$i.log" 2>&1 \
    || { echo "    采集失败: $(grep -E '拒绝|SafetyAbort|Error' "$O/pc$i.log" | tail -1)"; continue; }
  $PY scripts/apply_demonstrated_grasp_to_scenario.py "$O/p$i.yaml" \
    --row-templates "$ROW_TEMPLATES" --layer upper --output "$O/p$i.yaml" > "$O/pt$i.log" 2>&1 \
    || { echo "    行模板匹配失败: $(tail -1 "$O/pt$i.log")"; continue; }
  cd "$G/mtc_ws" && source_mtc_workspace
  rm -f "$O/p$i.json"*
  # A dropped /get_planning_scene response leaves the planner waiting forever
  # -- 2026-08-06 20:10 attempt 3 sat there until it was killed by hand, and
  # the retry loop above it never got its turn.  A planning attempt that has
  # not answered in four minutes is a failed attempt, not a slow one.
  timeout -k 10 240 \
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
cleanup_stack

# carry_home's target IS grasp_start_right_joints_deg -- the same pose
# normalize_right_arm drives to.  The difference was everything around it:
# normalize_to_grasp_start re-centres the head first, so it loads the YOLO
# archive and opens the RealSense to move an arm to a taught joint pose that
# needs no perception at all.  Measured 2026-08-07 on identical dry runs:
# 16 s against 1 s.  Both arms of the cycle retract, so it is twice that.
# The tuck evidence this used to stamp into the pick record now only feeds the
# lift's advisory block, which no longer gates anything.
say "阶段 2.5  持瓶回收拢位（无相机通路）"
cd "$G"
if $PY scripts/normalize_right_arm.py --skip-lift --speed 30 \
    --execute > "$O/tuck.log" 2>&1; then
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

start_stack place

say "阶段 4  放置（空位+场景+规划+执行 都在同一次尝试内）"
for i in 1 2 3 4 5; do
  echo "  --- 尝试 $i ---"
  cd "$G"
  # The y span used to be 0.58..0.72 -- 140 mm -- against delivery_table's
  # 100 mm table_edge_margin_m, which is subtracted from BOTH sides.  That left
  # arange(0.68, 0.62): empty, so every place attempt died on "no candidate
  # region" before looking at a single point.  2026-08-07 with 0.50..0.85 the
  # same scene produced two candidates immediately, at y=0.700 and y=0.740 --
  # note the second was outside the old window entirely.  The margin itself is
  # a table default (do not place near an edge you can push a bottle off); a
  # shelf bin's edges are panels, and its measured depth is close to 330 mm.
  $PY scripts/capture_empty_shelf_places.py --expected-lift-mm 250 \
    --product-code "$PRODUCT_CODE" \
    --roi-min -0.40 0.50 -0.45 --roi-max 0.23 0.85 0.05 \
    --lift-execution-record "$O/lift_record.json" \
    --output "$O/pl$i.json" > "$O/plc$i.log" 2>&1 \
    || { echo "    空位采集失败: $(grep -E '拒绝|SafetyAbort|Error' "$O/plc$i.log" | tail -1)"; continue; }
  $PY scripts/empty_shelf_places_to_mtc_scenario.py "$O/pl$i.json" "$O/pls$i.yaml" \
    --product-code "$PRODUCT_CODE" --row-templates "$ROW_TEMPLATES" > "$O/plg$i.log" 2>&1 \
    || { echo "    场景生成失败: $(tail -1 "$O/plg$i.log")"; continue; }
  cd "$G/mtc_ws" && source_mtc_workspace
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
