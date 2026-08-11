# 重开侧桌投放：操作者已给的量测，和还缺什么

**状态：入口是关着的，但关它的已经不是代码里那道硬拒。**

`d528562` 曾经在 `run()` 里对 `DeliverMode.DISPENSE` 无条件 `raise`。那是三道锁里最外面
的一道，而前两道本来就锁着：`load_safety_profile` 在 `enabled: false` 时直接 abort，
`--execute` 时还要求 `verified_for_execution: true`；之后 `_validated_shelf_ready` 对
`side_table_delivery` 里每个 null 字段和每个 `false` 的确认位逐项 fail-closed。硬拒唯一的
额外效果是**盖住了到底缺哪一项**，所以它已被删除，五个验收测试也已取消 skip。

现在关着入口的是 profile 本身：`side_table_template` 的 `enabled` 和
`verified_for_execution` 都是 false，`side_table_delivery` 里还有 50 项没填。
重开 = 把数字填齐 + 现场逐项翻确认位，**不需要再改代码**。

随时可以看还缺什么，不用连机器人：

```bash
python3 scripts/side_table_profile_status.py
```

这份文档不是修复，是把操作者已经量出来的数字和它们各自的去处记下来，免得下次从零开始。

---

## 1. 操作者 2026-08-11 给出的量测

| 量 | 值 | 怎么来的 |
|---|---|---|
| 桌面高 | 800 mm | 操作者实测 |
| 桌长 | 680 mm | 操作者实测 |
| 桌宽 | 400 mm | 操作者实测 |
| 投放时升降高度 | **约 515 mm** | 操作者估计，**需现场确认** |
| 朝桌子转向 | 原地**逆时针** 90° | 操作者描述 |
| 朝货架转回 | 原地**顺时针** 90° | 操作者描述 |
| 桌子当前方位 | 机器人**左侧** | 操作者描述 |
| **底盘 yaw 符号** | **正 = 逆时针 = 左** | **实测，见下** |

### 底盘 yaw 符号：已实测，不再是推断

下发 `angular = +0.12 rad/s`，读到 `theta` 从 **−2.534 → −1.389**
（Δ = **+1.145 rad**，与 0.12 rad/s × ≈9.5 s 吻合）。

**正角速度 ⇒ theta 增大 ⇒ 逆时针。** 标准右手定则 / REP-103（Z 轴朝上，
俯视逆时针为正），与 `agv_turn_once.cpp` 里的注释一致：
「正=逆时针/左，负=顺时针/右」。

代码这一侧是同一套约定，不需要任何转换：`mobile_body.py::_body_transform`
构造的是标准 CCW 旋转矩阵 `[[c,-s],[s,c]]`，`ChassisAdapter.rotate_relative`
把 `yaw_rad` 原样传给旋转助手。

**结论：桌子在左侧 ⇒ `body_rotation_yaw_deg = +90.0`，转回货架 = −90.0。**
这一条不必再上机验证。（在此之前这份文档写的是「推断 +90，必须实测」——
实测早就做过，只是这个仓库里没记，所以又被当成未知项复述了一遍。）

---

## 2. 字段映射

`mobile_body.py::_validated_shelf_ready` 要求下面这些非空且有限，缺一项就 `SafetyAbort`。

| profile 字段 | 上面哪个量 | 状态 |
|---|---|---|
| `body_rotation_yaw_deg` | 逆时针 90° ⇒ **`+90.0`** | **已填**。`target_yaw = start.chassis.yaw_rad + radians(该值)`。符号已实测确认（§1 末尾），不是推断，不需要再点动验证 |
| `target_lift_height_mm` | 515 | **已填**，但那是操作者的估计值。落桌前必须按实际桌面高复核，`lift_transition_verified` 在复核前保持 false |
| `max_angular_speed_radps` | 0.12 | **已填**。不是现场量测，是 `woosh_rotate_relative.cpp` 里在这台底盘上跑通过的值，也是它自己的上限 0.20 之内 |
| `rotation_tolerance_deg` | 2.0 | **已填**。闭环收敛判据，助手内部上限 5° |
| `rotation_timeout_s` | 30.0 | **已填**。90° 在 0.12 rad/s 下光转就要 ≈13 s，加上末端 P 控制减速段（`|err|*0.7`，最低 0.03 rad/s）和每 200 ms 一次的位姿复查，实测余量按 30 s 留；助手允许 5–60 s。**转不动或转过头，先动这个旋钮** |
| `max_base_translation_m` | 0.035 | **已填**，助手的默认平移守卫。「原地」不是零，底盘会漂；实际生效值取它和 `shelf_ready.xy_tolerance_m` 里更严的那个 |
| `table_roi.min` / `.max` | — | **不能从 800/680/400 推出来**。ROI 是 `right_controller_base` 系下的三维盒子，需要桌子相对机械臂基座的**位置**，桌子的三个尺寸只给了大小。转身后现场量 |
| `source_lift_height_mm` | 货架侧取货高度 | 未给 |
| `transport_joints_deg` | 运输途中的手臂姿态 | 未给，需示教 |
| `shelf_ready.{x_m,y_m,yaw_deg,lift_height_mm}` | 货架前的底盘停位 | 未给，需实测 |
| `shelf_ready.{xy_tolerance_m,yaw_tolerance_deg,lift_tolerance_mm}` | 各自容差 | 未给 |
| `rotation_sweep.positive.clearance_m` | 逆时针转时整机扫掠的最小净空 | **未测**，卷尺活：量机器人扫掠圆周围最窄处 |
| `rotation_sweep.negative.clearance_m` | 顺时针转回时的最小净空 | **未测**，同上 |
| `body_lift_speed` / `target_lift_tolerance_mm` | 升降限值 | 未给。注意升降适配器自己的到位判据是 ±5 mm，容差不能比它更紧 |

七个确认位 —— `transport_pose_verified`、`shelf_ready_verified`、
`lift_transition_verified`、`table_roi_verified`、`workspace_verified`、
`keepouts_verified`、`bottle_tcp_verified` —— 以及
`rotation_sweep.{positive,negative}.verified`，
**只能在现场逐项确认后才置 true**。把数字填进去而把确认位留 false 是安全的；
反过来（先置 true 再补测）是这个 profile 存在的全部意义所在，不要做。

---

## 3. 数字填了，确认位一个没翻

已实测和已审定的六项（§2 里标「已填」的）现在就在 profile 里。
填数字而确认位留 false 是安全的：闸照样拒绝，但现场不用再从文档里抄一遍。
反过来（先翻确认位再补测）是这个 profile 存在的全部意义所在，不要做 ——
底盘旋转是这个项目里后果最大的动作，整机扫掠，不是一条手臂。

剩下的每一项现场量一项、确认一项。

---

## 4. 重开的顺序

先跑 `python3 scripts/side_table_profile_status.py` 看一眼当前缺口（离线，不用连机器人）。

1. ~~确认底盘 yaw 符号~~ —— **已完成**，见 §1 末尾：`+90.0` 朝桌，`-90.0` 回货架。
2. ~~旋转限值~~ —— **已填**，见 §2：速度/容差/超时/平移守卫取的是闭环助手在这台
   底盘上验证过的值。转不动或转过头先调 `rotation_timeout_s`。
3. 量正反两个方向旋转时整机扫掠的最小净空，填 `rotation_sweep.{positive,negative}`。
   **这是卷尺活，不需要工具** —— `5b66b56` 删掉的 `site_check.py` 是通用现场自检
   （时钟、模型资产、头部舵机、夹爪、点云流、规划彩排），它从没量过旋转净空，
   不用重建。（这份文档之前把它列成第一步，是错的。）
4. 示教 `transport_joints_deg`（运输姿态）和货架前的 `shelf_ready` 停位。
5. 转身后量 `table_roi`（桌子相对机械臂基座的位置，不是桌子的三个尺寸），
   用 `observe_output_table` 干跑一次，确认能在桌面上找出候选。
6. 量 profile 层的 `tcp_workspace` / `allowed_tcp_zones` / `keepout_boxes` ——
   转身后的手臂包络，现在还是 null，`workspace_verified` / `keepouts_verified`
   指的就是它们。
7. 逐项置确认位，最后翻 `enabled` 和 `verified_for_execution`。
   **代码这一侧不用再改** —— `run()` 里的硬拒已经删掉，验收测试已经取消 skip，
   `test_the_side_table_dispense_entry_is_closed_by_its_profile` 是那道绊线：
   真要重开，它会失败，你得有意识地删掉它。

---

## 5. 当前的替代路线

下层货架投放已经很近了：`92e24f2` 之后，门位往后的整条动作都能解，
横移那段的悬空碎块也过滤掉了（离线复核：瓶盒沿整条插值压到的体素 228 → 0），
只差一次真机验证。**在侧桌投放重开之前，放回货架是成本低得多的出货路径。**
