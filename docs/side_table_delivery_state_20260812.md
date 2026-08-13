# 侧桌投放现状（2026-08-12 收工）

给下一个接手的人（含 codex）。**一句话：代码通了，一次真机都没跑过。**

前一份 `side_table_delivery_reopening.md` 讲的是"怎么重开"，已过时——入口开了。
这份讲"现在是什么状态、哪些数是真的、哪些是编的、下一步动什么"。

---

## 1. 一条命令

```bash
DISPENSE=1 SHELF_LAYER=auto TARGET_PRODUCT=P01 \
SAFETY_PROFILE=shelf_template DELIVERY_SAFETY_PROFILE=side_table_template \
COMMISSIONING_SPEED=30 bash scripts/run_task.sh from-start
```

在 **Mac** 上跑，它自己 ssh 到机器人。`ssh -tt`，日志实时回显在你的终端。
跑之前先 `python3 scripts/robot_code_drift.py --push`。

`SHELF_LAYER` 三种值：`auto`（搜索）、具体高度如 `250`（跳过搜索）、不设（用
profile 自己的 SHELF_READY 高度 647）。`auto` 必须同时给 `TARGET_PRODUCT`——
搜"随便什么瓶子"会停在第一个有东西的层，那不是在找你要的货。

`SAFETY_PROFILE` 只能是 `shelf_template`：`table_demo` 的
`verified_for_execution` 是 false，`--execute` 直接拒；而且
`_validate_side_table_profile_pair` 要求源和投放 profile 的 `home_joints_deg`
与 `tool_mount_calibration` 全等，只有它对得上。

---

## 2. 完整链路

```
层间搜索（shell，机器人端）        scripts/find_product_shelf_layer.py
  └ 遍历 search_lift_heights_mm = [647, 250]
    空载升降到该层 → 头部检测 P01 → 找到就打印高度、退出 0
    每层都没有 → 退出 3（补货，不是故障）
    相机/深度/升降/profile 出错 → 退出 1/2（不换层，不重试）
        ↓ --shelf-layer-lift-mm <高度>
SHELF_READY 捕获（arm 会话之前）    orchestrator._capture_shelf_ready_for_dispense
  └ _side_table_config() 把选中的层写进 shelf_ready.lift_height_mm
    和 source_lift_height_mm（两者必须一起动，见 §5）
initialize() → 预检 → 头锁 → 抓取 → 抬升
携瓶收进运输姿态                    _move_held_to_transport_posture
升降 该层高度 → 515                 mobile_body.position_for_delivery
闭环原地旋转 +90°                   scripts/woosh_rotate_relative.cpp
转后重建桌面点云 → 选放置点          delivery_table.observe_output_table
降到站位高度（MoveIt + 稠密围栏复核）
腕部核查落点净空                    _assert_landing_patch_clear_from_wrist
视觉伺服落桌                        _servo_bottle_onto_table
松爪 → 退开 → 三维确认释放
右臂回 home → ReturnAuthorization
闭环旋转 −90° → 升降回该层 → 校验回到 SHELF_READY
```

---

## 3. 这次做的四件事，和为什么

### 3.1 放置高度改成从商品表推导

**之前会把瓶子怼进桌面 68mm。** profile 里手填的
`bottle_bottom_below_tcp_m = 0.04` 原本是碰撞包络参数（那里错了无所谓，包络是
个粗圆柱），被当成了放置高度。P01 实际是 `height_m/2 = 0.1075`。

现在 `_bottle_bottom_below_tcp_m()` 从 `product_geometry(code)["height_m"]/2`
推。这是全仓库既有的约定——`empty_shelf_places_to_mtc_scenario` 放瓶心在
`surface_z + height_m/2`，携瓶碰撞扣除也是绕同一点量 `|dz| <= height_m/2`。

profile 里那个字段还在（loader 要求），但**放置不再读它**。

### 3.2 放置视觉伺服（头部 + 腕部）

即使推导对了，还有约 30mm 不确定度：`tool_mount_calibration` 的
`provenance` 是 `nominal_functionally_validated`——link7→flange 17.2mm、
flange→TCP 151mm 都是**标称值**，真实误差靠 `grasp_stop_short_m = 0.030`
吸收，而那是为货架的**水平进入方向**调的，自上而下不适用。它自己的
evidence_id 就写着「do not transfer to another approach direction, or to a
geometric grasp-point derivation」。

所以闭环量不是 TCP 的 z，是**头部实测的 TCP→瓶底偏移**——把标称链直接踢出回路。

- `_measure_held_bottle_bottom_z`（头部）：TCP 周围按商品半径开圆柱，取桌面以上
  TCP 以下的点，2% 分位数当瓶底（不用最小值：一个杂散像素不该决定何时松爪），
  多帧中位数。
- `_assert_landing_patch_clear_from_wrist`（腕部）：**只在站位高度看一次**。
  D435 最近测 0.12m，最后几厘米腕部相机在自己盲区里，那时报"落点干净"是必然的，
  当净空用比不看更糟。
- `_servo_bottle_onto_table`：分离度 > 25mm 时边降边收敛偏移估计，之下停止测量、
  用已收敛值提交最后一段。**为什么不一路伺服到接触**：瓶底和桌面在点云里会合并，
  间隙测量朝零塌陷——在最需要精度处最自信地错。

守卫：实测偏移与商品表差 > 35mm 中止；单步 ≤ 12mm；累计 ≤ 120mm；最后提交段超
单步上限中止；算出瓶底已低于目标中止（绝不用"继续下降"回答"已经太低"）。

证据落在 `<run_dir>/output_place_servo.json`。

### 3.3 层间搜索

codex 做的在 `run_cross_layer_cycle.sh` 里（MTC 抓取 → 放回下层货架）。侧桌是
另一条流水线，得单独接。

**为什么搜索在 shell 而不在 Python 里**：DISPENSE 流程在 `initialize()`
**之前**捕获不可变的 SHELF_READY 快照，就是为了让任何 arm/gripper 命令都不能
先于它；而检测器是 `initialize()` 才建的。要塞进那个窗口就得拆开一个按安全顺序
编排的函数，风险比功能本身大。升降是 body-only 动作，所以搜索放在任务之前。

顺手修了 codex 版本里一个真 bug：**相机挂了会被误判成"这层没货"**。采集循环取不
到帧就 `continue`，`detector_hits` 永远 0 → 抛 `BottleDetectionLost` → shell
当空层换层 → 报"两层都没有"。人会去翻货架，其实是相机死了。现在 `localize` 里加
了 `frames_seen`，零帧抛 `CameraFrameUnavailable`（不是 `BottleDetectionLost`
的子类，采集脚本的 handler 不接，退出码 1）。回归测试
`test_a_dead_camera_is_never_reported_as_an_empty_shelf`。

### 3.4 profile 填满并启用

操作者 2026-08-12 明确要求，**未做完整现场勘测**。见 §4。

---

## 4. 哪些数是真的，哪些是编的

`side_table_template.description` 里逐条写了同样的内容——**改 profile 时一起
更新它，那是以后唯一能分辨的线索。**

| 来源 | 字段 |
|---|---|
| **今天从机器人读的** | `shelf_ready.{x_m,y_m,yaw_deg,lift_height_mm}` = −0.333925 / 0.15595 / −67.2237° / 647；`source_lift_height_mm` = 647 |
| **复用仓库已有值** | `transport_joints_deg` = `shelf_template.grasp_start_right_joints_deg`（阶段 2.5 已验证的持瓶收拢位）；瓶体几何 4 项取自 `DemoParams`；`tcp_workspace` / `allowed_tcp_zones` 沿用货架已测包络（同臂同工具，坐标系随臂走）；`tool_mount_calibration` 逐字节复制 |
| **实测过但符号级** | `body_rotation_yaw_deg` = +90（yaw 符号实测：`angular=+0.12` 时 theta −2.534 → −1.389，正 = 逆时针 = 左） |
| **闭环助手的审定值** | `max_angular_speed_radps` 0.12 / `rotation_tolerance_deg` 2.0 / `max_base_translation_m` 0.035；`rotation_timeout_s` 给 30 而非默认 25（90° 在 0.12 rad/s 下光转 ≈13s，加末端 P 减速段和每 200ms 位姿复查，25 贴边） |
| **编的** | `rotation_sweep` 净空 0.30（`_assert_sweep` 分辨不了实测和手输，真正保护是人站着看 + 急停）；`table_roi`；桌面拟合和放置参数 16 项 |
| **不能为空所以编的** | `keepout_boxes` 一个地板禁区。**没复用货架那三个**——它们描述货架本身，转身后不在那儿，而且正好压在桌面 ROI（y 0.54–1.5）上，会挡掉每个候选 |

### `table_roi` 是最该先收紧的

它不是"桌子在哪"，是"去哪儿找"。但 `_fit_one_height` 的做法是：ROI 内的点按
5mm 分箱，**取点数最多的那一箱**当桌面。没有任何"这是不是桌子"的判断——不看
法向、不看尺寸、不看连通性。

我给的 z 跨度是 700mm，升降 515 时**很可能同时框进地面**，而地面的点通常远多于
一张 680×400 的桌子。多帧一致性检查（`spread > 15mm`）抓不到这个：地面每帧都
一致，拟合得又稳又漂亮，只是拟合的是地面。

还有两个连带影响：候选网格从 ROI 边界生成；`table_edge_margin_m`（60mm，本意是
离桌沿留余量）也是从 ROI 边界扣的——ROI 比桌子大一圈，这个边距**等于没有**，
桌沿保护只剩 `table_support_radius_m`。

**第一次跑转过去之后，看日志报的拟合桌高。** 合理就把 z 窗口收到 ±100mm；
明显是地面（比预期低一截）就停，别落瓶。

---

## 5. 改代码前必须知道的两条约束

**SHELF_READY 是不可变锚点。** 它在 arm 会话存在之前捕获，返程要
`_assert_matches_start` 校验回到同一状态。所以选层必须发生在捕获之前，而且
`shelf_ready.lift_height_mm` 和 `source_lift_height_mm` **必须一起改**——
`_shelf_ready_limits` 会拒绝两者不一致的组合。`_side_table_config()` 就是干
这个的，全流程 8 个读 config 的地方都走它，不要再直接读
`self.delivery_safety.side_table_delivery`。

**`--shelf-layer-lift-mm` 只能选，不能造。** 值必须在 profile 的
`search_lift_heights_mm` 里，第一项必须等于 profile 自己的 SHELF_READY 高度
（无搜索时的默认层）。加新层改 profile，不要改代码。

---

## 6. 没做的

- **侧桌那条没有 MTC 抓取**。`run_task.sh` 用的是 head/wrist 定位 +
  `_grasp_and_lift`；`run_cross_layer_cycle.sh` 用 MTC。两条抓取实现不同，
  没合并。
- **`_deliver_to_output`（orchestrator）没有生产调用方**，只剩自己 6 个测试
  养着。是更老的定关节投放（bin/取货口），`output_joints_deg` 在所有 profile
  里都是 null。删或留，没定。
- **`grasp_stop_short_m = 0.030` 那 30mm 到底是什么，仍然没量过。** 视觉伺服
  绕开了它，没有解决它。第一次跑完，`output_place_servo.json` 里的
  `measured_offset_m` 减 `catalog_offset_m` 就是它在垂直方向的真值——**记下来。**
- **底盘原地转 90° 的平移漂移没有数据。** `max_base_translation_m = 0.035`
  超了就 `SafetyAbort`，而那时瓶子已经放在桌上、机器人转着 90° 卡在那儿。
  空载转 90° 再转回、重复 10 次记 x/y 漂移，是最该单独做的实验。
- **转向中止后没有软件恢复路径。** `position_for_delivery` 的
  `except BaseException` 清掉 `_captured_shelf_ready`，之后 `return_to_shelf_ready`
  必然拒绝。本项目遥操全程关闭，所以剩下的大概只有物理旋钮打到 `kManual` 人力推
  ——**这条恢复流程没有写在任何地方，应该补。**

---

## 7. 跑的时候盯这两行

```
桌面放置视觉闭环观测  第 N 次：瓶底距桌面 XX mm，实测 TCP-瓶底偏移 YY mm（商品表 107.5 mm）
```

`YY` 与 107.5 的差就是标称工具链在垂直方向的真实误差。差 > 35mm 自动中止。

```
底盘原地旋转完成  实时 yaw=...°，x/y=(...,...)；无平移指令
```

x/y 相对 SHELF_READY 漂了多少，就是 §6 那个从来没有的数据。

---

## 8. 测试

`PYTHONPATH="$PWD" pytest test/ -q` → **691 passed**（2026-08-12）。

这次新增/改写的：

- `test_place_visual_servo.py`（5 个）——伺服按实测偏移落桌；抓取点偏 30mm 时
  信目录会怼进桌面、信实测不会；步长有界；偏移离谱中止。
- `test_shelf_layer_resolution.py`（8 个）——选层同时改 admission 和返程；不
  篡改原 profile；未列出的高度拒绝；`search_lift_heights_mm` 格式校验。
- `test_a_dead_camera_is_never_reported_as_an_empty_shelf`——相机故障不是空层。
- 四个"侧桌关着"的绊线测试全部改写成断言新状态。其中
  `test_..._declares_its_provenance` 现在钉的是 **description 里必须继续写明
  哪些数没量过**。
