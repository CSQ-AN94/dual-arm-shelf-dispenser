# 给接手这个仓库的每一个会话

这份文件会被自动读到。**下面每一条都是踩过坑换来的，不是风格偏好。**

---

## 1. 抓取主线只有一条

```
scripts/run_cross_layer_cycle.sh
```

八步：双臂+升降原子归位 → 只规划 MoveIt 栈 → 夹爪空夹标定 →
**固定头部相机**采集场景 → 122 个示教点的行模板匹配 → MTC 规划 → 执行 → 收臂。

完整定义和产物对照表在 **[`docs/GRASP_MAINLINE.md`](docs/GRASP_MAINLINE.md)**，
**改抓取相关代码之前先读它。**

**主线里没有的东西**（因为旧管线有，很容易搞混）：

- **没有右腕相机、没有观察位。** 定位从头到尾只有固定头部相机
- **没有第二个 home。** 只有一个示教停放位姿：双臂 `grasp_start`（右臂+左臂+升降 647）

`scripts/run_task.sh` 是**已废弃**的 table_demo 时代管线，直接运行会拒绝。
不要"顺手修一下让它能跑"——它 2026-08-12 四次尝试没有一次走到抓取。

## 2. 开工先确认自己在哪条分支

```bash
git status -sb && git log --oneline -3
```

**这个仓库同时被多个会话动过，分支会在你脚下切换。** 2026-08-12 就发生过：
一个会话把十个提交做在 `side-table-fill-profile` 上，另一个会话在 `main` 上重写
README，两边都对但互相看不见。当时的症状是"测试数从 734 掉到 715"——
不是测试变了，是分支换了，少的 19 条正是另一边刚加的。

看到测试数、文件或行为莫名其妙变化时，**先怀疑分支，再怀疑代码**。

## 3. 头部有一个固定基准角，任务全程不许动它

```python
HEAD_REFERENCE = {"angle1": 398, "angle2": 516}   # 俯仰最低、偏航居中
TOLERANCE = 5
```

（`shelf_dispenser/head_lock.py`，2026-07-08 标定会话实测。）

**整条抓取主线的定位全靠这个固定头部相机**，所以头一动，这一轮采集的所有
三维点就都错了。`capture_mtc_direct_pick_scene.py` 会在采集前校验，
不在基准角就直接拒绝、并且**不会自动去转它**。

两条光看数字不会知道的坑：

- 厂商服务把 `angle1` 的命令下限卡在 **400**，而反馈在机械限位附近读到的是 **398±几**
- **厂商方向键固定 50 步长，到不了 `angle2=516`** —— 只能用绝对位置命令，
  也就是 `scripts/head_position_lock.py restore`，不要用方向键去凑

检查 / 复位：

```bash
python3 scripts/head_position_lock.py check      # 只读
python3 scripts/head_position_lock.py restore    # 绝对位置命令 + 闭环复核
```

⚠️ 这台机器的 **pitch 舵机（ID=1）有已知问题**，相机上下动不了、只能左右动。
所以 `angle1` 实际上就停在机械限位上，别指望调它。

## 4. 代码在机器人上跑，Mac 上跑不了臂

改完必须先推，否则机器人跑的还是旧代码：

```bash
python scripts/robot_code_drift.py --push
```

- **新脚本要自己加进 `scripts/robot_code_drift.py` 的 `TRACKED`**，
  否则不会被推上去，而且照样打印"已同步"
- 它**只推不删**，机器人上的旧文件不会消失
- **C++ 改过必须重新编译**，不编译不生效：
  `colcon build --packages-select grabber_mtc_planner`

## 5. 遥操全程关闭

**永远不要**恢复遥操，不要跑 `upstart_all.sh`，不要重启 `atom` / `zhixing_ctrl.py`。
这是这个项目的硬约束。

## 6. 数值必须带证据，确认位不许顺手翻

安全相关的常量带 `evidence_id`，指向测出它的那次运行。

- 把数字填进 profile、而确认位（`*_verified`）留 `false`：**安全**
- 填了数字顺手把确认位翻成 `true`：**这个仓库唯一明令禁止的事**

未实测的值要在文档里写明"这是手输的"。当前的例子是侧桌投放的净空参数，
见 [`docs/side_table_delivery_state_20260812.md`](docs/side_table_delivery_state_20260812.md)。

## 7. 不要制造第二份副本

这个仓库在"同一个量算了两遍"上栽过两次（一次是双 home，一次是行位映射）。

需要一个已经存在的量时，**调用产生它的那个函数**，不要重新推导。
例如行位→槽位必须走 `apply_demonstrated_grasp_to_scenario.select_row_candidate`。

## 8. 判断成败看证据文件，不看日志最后一行

抓取成功的判据是 `CYCLE_OUT/pick_record.json` **存在**。

2026-08-11 有一次执行已经成功、记录已落盘，但脚本被中断、横幅没打出来，
被误判成失败，接着又白跑了两轮。

## 9. 中止是常态，不是异常

安全链设计成 fail-closed：执行审计会否掉九成的规划解（成功那几轮也只活下来 2~6 条），
围栏和控制器队列上限会在下发运动**之前**拒绝。

看到"未解出"先看 `p*.json` 的 `complete_solution_count_by_arm`：
有完整解但全被否 = 审计拒了（看 `pp*.log` 的 `rejecting ... solution:` 哪几项 `false`）；
零完整解 = 规划没解出来（看 `earliest_failure_stage_by_arm`）。

## 10. 上机之前

操作手册是 **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)** —— 开机三条、复现抓取、
跨层搜索、整套流程的推进顺序和验收标准，都在里面。

**先量，再下结论。** 这个项目里绝大多数"看起来像 A 的问题"最后都是 B，
而每次都是靠日志和产物定位的，不是靠推理。

## 11. 测试

```bash
python -m pytest test/ mtc_ws/src/grabber_mtc_planner/test -q
```

不需要机器人、ROS 或相机。改动落地前必须全绿。
