# 抓取主线（2026-08-12 定稿）

**这份文档定义什么是主线。** 不在这里的抓取路径都不是主线，
新代码只往这条上加，不要再开第二条。

判定依据不是谁写得新，是**真机成绩**：主线这条 2026-08-11 四次抓取三次成功
（外星人上层 0.26、柠檬茶上层 0.54、阿萨姆下层 0.56），
证据在 `outputs/runs/20260811_picks/`——目录里有哪些产物就等于跑过哪些步骤。

---

## 1. 主线是这 8 步

启动器：**`scripts/run_cross_layer_cycle.sh`**

```
PRODUCT_CODE=P01 LAYER=upper PICK_ONLY=1 bash scripts/run_cross_layer_cycle.sh
```

| # | 步骤 | 入口 | 产物 |
|---|---|---|---|
| 0 | 右臂 + 升降预归位 | `normalize_to_grasp_start.py --right-and-lift-only` | `norm_right_lift.log` |
| 0.5 | 左臂归位 | `normalize_left_arm.py` | `norm_left.log` |
| 1 | **双臂 + 升降原子门禁** | `normalize_to_grasp_start.py` | `norm.log` |
| 1.5 | （下层才有）空手升降到 250 | 内联 | `lift_down.log` |
| — | 起只规划 MoveIt 栈 | `live_state_plan_only.launch.py` | `moveit_stack_pick.log` |
| 2a | 夹爪空夹标定 | `calibrate_mtc_gripper.py` | `grip.json` |
| 2b | **只用固定头部相机**采集场景 | `capture_mtc_direct_pick_scene.py` | `p*.yaml` |
| 2c | 行模板匹配（122 个示教点插值） | `apply_demonstrated_grasp_to_scenario.py` | `pt*.log` |
| 2d | MTC 规划 | `plan_shelf_transfer_experimental.launch.py` | `p*.json` |
| 2e | 执行抓取 | `execute_mtc_trajectory.py pick` | `pick_record.json` |
| 2.5 | 收臂回收拢位 | `normalize_right_arm.py --skip-lift` | `tuck.log` |

**主线里没有的东西**（重要，因为旧管线有）：

- **没有右腕相机、没有观察位。** 定位从头到尾只有固定头部相机，
  `capture_mtc_direct_pick_scene.py` 明写着 "never connects either arm"。
  `grep -rl "右腕\|观察位\|wrist" outputs/runs/20260811_picks/pick_p01/*.log` → 零命中。
- **没有第二个 home。** 全项目只有一个示教停放位姿：双臂 `grasp_start`
  （右臂 + 左臂 + 升降 647，原子门禁整体校验）。
  `home_joints_deg` 已于 2026-08-12 从所有 profile 和 `SafetyProfile` 删除。

## 2. 主线还缺的两块（尚未接上）

### 2.1 跨层搜索 —— 已验证，但接在错的启动器上

`scripts/find_product_shelf_layer.py` 遍历 `search_lift_heights_mm = [647, 250]`，
空载升降到每层用头部相机找目标，找到就打印 `SHELF_LAYER_LIFT_MM=<高度>`。
**2026-08-12 第一次真机运行就成功**（647 层 0/7 → 降到 250 → 7/7 共识、散布 3.1 mm）。

**问题**：它只被 `run_task.sh`（旧管线启动器）调用，
主线的 `run_cross_layer_cycle.sh` 里一次都没有——主线现在靠人手写 `LAYER=upper|lower`。

**要做**：给 `run_cross_layer_cycle.sh` 加 `LAYER=auto`，调它拿到高度再往下走。
一个已知的接口毛刺：它的搜索高度表读自 `side_table_template` profile，
而货架抓取本不该依赖投放 profile——接的时候要么把高度表搬到 `shelf_template`，
要么显式说明这个依赖。

**"几中几"已定（2026-08-12 修）**：判据原本只看检测命中数是不是 0，
所以一帧误检（`1/7`）就被判成故障、整个搜索中止。现在按共识下限 `3` 切三段：

| 可用三维点 | 判定 | 理由 |
|---|---|---|
| `0` 且检测命中 > 0 | **故障** | 检测到了但深度一帧都没产出 —— 深度坏了，不是货架空（2026-07-18 真实运行 pin 住） |
| `1 ~ 2` | **这层没有** | 低于共识下限，无论如何都不可能确认成功，属于杂散帧 |
| `3 ~ 6` | **故障** | 足够多的帧认同有东西却定不下来，值得停下来看，而不是悄悄跳过一个有货的层 |
| `7 / 7` | 通过 | 实测成功的两次都是 7/7，散布 1.8~3.1 mm |

分界 `3` 不是拍的：共识判据自己的下限就写着 `max(3, ceil(0.70·N))`。

### 2.2 数据库 —— 写好了，零调用者

`shelf_dispenser/inventory.py`，9 个测试通过，**没有任何代码用它**。

它的 API 正好是需要的形状：

```python
observe(inv, "B2", occupied=True, contents="P06", scene_version=...)
find(inv, "P06", max_age_s=...)     # -> ["B2"]
record_pick(inv, "B2")
```

槽位 A1–A4 / B1–B4 不是编的，是从 122 个示教点推出来的
（每层 61 点、10 mm 间距、600 mm 行长切四格，每格能被哪条臂够到直接读 `eligible_arms`）。

**要做的三处接线**：

1. 层间搜索每扫一层，把**看到的全部**写进去（现在只用"有没有目标"就把其余全扔了）
2. 抓取成功后 `record_pick()`
3. 放置成功后 `record_place()`

**接之前要先解决的一件事**：记"位置"需要把检测点映射到槽位，
而这个映射必须复用 `apply_demonstrated_grasp_to_scenario.select_row_candidate`
（按示教点的 MoveIt X 找最近行位），**不能自己再算一遍**——
这个仓库已经在"第二份副本"上栽过两次。

**还有一条设计约束**：库存不是世界的真相，是"机器人上次看到的东西 + 时间戳"。
人随手挪一瓶它就错了。所以 `find()` 强制要求调用方给 `max_age_s`，
而且**命中之后没找到必须能退回去重新扫**，不能直接失败。

## 3. 已废弃的管线：`run_task.sh`

**它现在跑不起来了**（2026-08-12）。直接运行会拒绝并指向主线：

```
拒绝: 这条抓取管线已于 2026-08-12 废弃，不要运行。
  货架抓取请用主线：
    PRODUCT_CODE=P01 LAYER=upper PICK_ONLY=1 bash scripts/run_cross_layer_cycle.sh
```

做迁移时才用 `I_KNOW_THIS_PIPELINE_IS_RETIRED=1` 显式绕过。

**它是什么**：table_demo 时代的管线，头部粗定位 → **右腕观察位** → 右腕精定位 →
抓取 → 放置。2026-08-12 四次尝试全部没走到抓取，其中两次直接死于它自己的遗产
（228° 的 table_demo 停放位姿、34 个右腕观察位候选全淘汰），
同期主线真机抓取 4 次 3 成。

**停用它没有丢掉任何已验证的能力** —— 侧桌投放曾经只活在这条上，但它一次都没成功过。

### 3.1 侧桌投放怎么迁到主线（比看起来容易）

清点过了，**桌面观测的能力不在旧 orchestrator 里**：

| 需要的东西 | 在哪 | 状态 |
|---|---|---|
| 桌面拟合 + 候选点 | `shelf_dispenser/delivery_table.py`（库） | ✅ 现成，而且**主线的 `capture_empty_shelf_places.py` 已经在用它** |
| 采集空位 → 放置记录 | `scripts/capture_empty_shelf_places.py` | ✅ 主线脚本 |
| 记录 → MTC place 场景 | `scripts/empty_shelf_places_to_mtc_scenario.py` | ✅ 主线脚本 |
| MTC place 规划 + 执行 | `plan_shelf_transfer` place 段 / `execute_mtc_trajectory.py place` | ✅ 现成 |
| 底盘转身 | `shelf_dispenser/mobile_body.py` | ✅ 库，但**从未真机执行** |
| 落桌视觉伺服 | `orchestrator._servo_bottle_onto_table` | ⚠️ **只在旧 orchestrator 里**，要搬出来 |

**所以迁移 = 主线的 pick 之后，把 ROI 从「下层货架空位」换成「桌面」，中间插一段底盘转身。**
真正要搬的只有落桌伺服那一段。

**顺序**：
1. 先在**不转身**的情况下打通"MTC place 到桌面"（把桌子摆在机器人正前方够得到的地方），
   这样底盘旋转和放置两个未验证项不会同时上场
2. 通了再加转身，且**先空手单独验一次旋转**

## 3.2 一层四个点位：视觉先验（你提的，成立）

槽位不是新概念，`inventory.py` 里已经按 122 个示教点推出来了：每层 600 mm 切四格，
`A1–A4` / `B1–B4`，每格能被哪条臂够到直接读 `eligible_arms`。
而示教的那 8 条放置路径正好是每层 4 个点位。

**所以视觉不必从零找位置**：检测到目标之后，
`inventory.slot_for_row_position(layer, row_m)` 就能说出它在哪一格，
而那一格的示教点给出这一格的大致位姿和可达臂。这既是库存要记的"位置"，
也是给规划的先验。

**接线要注意**：行位映射必须复用
`apply_demonstrated_grasp_to_scenario.select_row_candidate`（按示教点的 MoveIt X
找最近行位），**不能自己再算一遍**——这个仓库已经在"第二份副本"上栽过两次。

## 3.3 其余不是主线的东西

| 路径 | 是什么 | 怎么处置 |
|---|---|---|
| `replay_demonstrated_trajectory.py` | 示教轨迹回放，真机验证过 4 次 | 保留，规划不出来时的兜底 |
| 各种 `mujoco_*.py`、标定、测量、诊断脚本 | 独立工具 | 保留，人工调用 |

**2026-08-12 已删除**（"旧的失败的"，它们自己的文档字符串就承认了）：

- `demonstrated_cross_layer_cycle.py` —— *"Fourteen consecutive autonomous attempts have failed"*
- `continue_manual_grasp.py` —— 从人手放进夹爪的抓取继续
- `diagnose_left_arm_kinematics.py` —— 左臂运动学一次性诊断，那个 bug 已修
- `run_task_autonomous.sh` / `run_task_resume.sh` / `start_task.sh` —— 三个 `exit 2` 的空壳，
  而且每次跑任务都在被 rsync 到机器人

## 4. 一句话

**主线 = `run_cross_layer_cycle.sh` 那 8 步（已验证）+ 跨层搜索（已验证但没接过来）
+ 库存（写好但零调用者）。** 前者是真的，后两者是待接。
在它们合成一条之前，"主线"这个词指的是第 1 节那 8 步。
