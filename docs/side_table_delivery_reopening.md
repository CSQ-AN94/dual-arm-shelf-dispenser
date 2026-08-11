# 重开侧桌投放：操作者已给的量测，和还缺什么

**状态：入口是关着的。** `d528562` 主动关闭了 `DeliverMode.DISPENSE`，五个测试被 skip
保留成重开的验收标准；`5b66b56` 删掉了产生现场量测的核查工具。
`shelf_dispenser/safety_profiles.json` 里 `side_table_template.side_table_delivery`
**每一个字段都是 null**，七个 `*_verified` 全是 false。

所以「把旋转方向改一下」这句话在当前代码上不成立 —— 没有旋转方向可改，那一栏是空的。
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

---

## 2. 字段映射

`mobile_body.py::_validated_shelf_ready` 要求下面这些非空且有限，缺一项就 `SafetyAbort`。

| profile 字段 | 上面哪个量 | 备注 |
|---|---|---|
| `body_rotation_yaw_deg` | 逆时针 90° | `target_yaw = start.chassis.yaw_rad + radians(该值)`。ROS 惯例 yaw 绕 +Z **逆时针为正**，所以推断值是 `+90.0` —— **但必须现场用小角度（比如 10°）实测确认底盘 yaw 的上报方向**，反了的话 `+90` 会朝货架转过去 |
| `target_lift_height_mm` | 515 | 操作者估计值，落桌前要按实际桌面高复核 |
| `table_roi.min` / `.max` | 800 / 680 / 400 | ROI 是**安全包络**，不是投放点。投放点由 `observe_output_table` 在现场点云里重算，ROI 只圈定它能在哪儿找 |
| `source_lift_height_mm` | 货架侧取货高度 | 未给 |
| `transport_joints_deg` | 运输途中的手臂姿态 | 未给，需示教 |
| `shelf_ready.{x_m,y_m,yaw_deg,lift_height_mm}` | 货架前的底盘停位 | 未给，需实测 |
| `shelf_ready.{xy_tolerance_m,yaw_tolerance_deg,lift_tolerance_mm}` | 各自容差 | 未给 |
| `rotation_sweep.positive.clearance_m` | 逆时针转时整机扫掠的最小净空 | **未测** |
| `rotation_sweep.negative.clearance_m` | 顺时针转回时的最小净空 | **未测** |
| `max_angular_speed_radps` / `rotation_tolerance_deg` / `rotation_timeout_s` | 旋转限值 | 未给 |
| `max_base_translation_m` | 原地旋转允许的平移漂移上限 | 未给。「原地」不是零，底盘会漂 |
| `body_lift_speed` / `target_lift_tolerance_mm` | 升降限值 | 未给 |

七个确认位 —— `transport_pose_verified`、`shelf_ready_verified`、
`lift_transition_verified`、`table_roi_verified`、`workspace_verified`、
`keepouts_verified`、`bottle_tcp_verified` —— 以及
`rotation_sweep.{positive,negative}.verified`，
**只能在现场逐项确认后才置 true**。把数字填进去而把确认位留 false 是安全的；
反过来（先置 true 再补测）是这个 profile 存在的全部意义所在，不要做。

---

## 3. 为什么没有直接把数字写进 profile

写进去而确认位是 false，闸照样拒绝，等于没写；
写进去顺手把确认位翻成 true，就把一整套现场校验绕过去了 ——
而底盘旋转是这个项目里后果最大的动作（整机扫掠，不是一条手臂）。

数字记在这里，现场标定时照着填，填一项确认一项。

---

## 4. 重开的顺序

1. **重建被删的现场核查工具**（`5b66b56` 删掉的那个），或者用等效办法测出
   正反两个方向旋转时的最小净空。没有这两个数，`_validated_shelf_ready` 第一关就过不去。
2. **确认底盘 yaw 符号**：小角度点动，读 `chassis.yaw_rad` 变化方向，再决定
   `body_rotation_yaw_deg` 填 `+90` 还是 `-90`。
3. 示教 `transport_joints_deg`（运输姿态）和货架前的 `shelf_ready` 停位。
4. 填 `table_roi`，用 `observe_output_table` 干跑一次，确认能在桌面上找出候选。
5. 逐项置确认位，最后重开 `DeliverMode.DISPENSE`，
   跑那五个被 skip 的测试当验收。

---

## 5. 当前的替代路线

下层货架投放已经很近了：`92e24f2` 之后，门位往后的整条动作都能解，
横移那段的悬空碎块也过滤掉了（离线复核：瓶盒沿整条插值压到的体素 228 → 0），
只差一次真机验证。**在侧桌投放重开之前，放回货架是成本低得多的出货路径。**
