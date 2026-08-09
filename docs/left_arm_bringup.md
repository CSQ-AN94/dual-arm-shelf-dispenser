# 左臂打通：从"只能规划"到"能抓一次"

**现状**：左臂的规划通路、围栏、进程隔离、示教位形**全部就绪**，唯一的闸是
`dual_rm75_arms.yaml` 里的 `execution_eligible: false`，
原因写在同一个文件里：`execution_block_reason: LEFT_TOOL_CALIBRATION_REQUIRED`。

**这份文档要做的事**：把那一行翻成 `true`，并且让它翻得有依据。

**读者**：接着做这件事的人。不需要读过之前的对话。

---

## 1. 为什么它被锁着

左臂的工具变换在 `mtc_ws/src/grabber_mtc_planner/config/dual_rm75_arms.yaml` 里是这样：

```yaml
- arm_id: left_arm
  ik_link: l_link7
  # Mirrored from the right arm so left-arm planning is possible at all.  It is
  # NOT a measurement, which is exactly why the arm is execution-blocked.
  tcp_transform_from_ik_link:
    [[1,0,0,0],[0,1,0,0],[0,0,1,0.1682],[0,0,0,1]]
  execution_eligible: false
  execution_block_reason: LEFT_TOOL_CALIBRATION_REQUIRED
```

`safety_profiles.json` 里一致：`left_tool_mount_calibration` 的
`provenance` 是 **`nominal_unvalidated`**，证据字段写的是"沿用右臂未经改动的
RMG24 标称链"。

**两条臂的夹爪是同型号、同装法，所以抄过来的值很可能是对的。**
锁着不是因为认为它错，是因为**没人验证过**。

这个仓库在"抄了没核对"上已经栽过两次，都不是小事：

- 升降的示教位姿是从别处抄的副本，J6 距控制器限位只剩 1.6°，落在 3° 安全余量带内，
  **指向它的规划一次都不可能成功**，两天后第一次真正规划左臂时才暴露；
- 同一个"体素立方体到瓶轴"的几何在两个脚本里写了两遍，一处用 `voxel/2`、
  一处用 `voxel/√2`，小的那个让抓取连挂五次。

所以下面第 2 步不是形式主义。它花十几分钟，能把"抄的"变成"量过的"。

---

## 2. 先做这一步：实测左臂工具链

用 `scripts/verify_dual_arm_by_touch.py`。**纯运动学，不用相机、不用棋盘格**，
因此独立于视觉标定的一切假设。

原理：让两条臂的末端**依次触碰同一批真实物理点**（螺丝钉尖、桌面记号都行），
每碰一个点就用只读连接分别记录两条臂各自汇报的末端位置。两条臂用什么关节姿态
够到这个点完全不重要，只要末端位置对得上。采集 **≥3 个不共线的点**后，
用 Kabsch 做刚体配准，直接解出两条臂基座之间的变换，
再与 `scripts/compose_dual_arm_transform.py` 的结果比对。

**操作要点**

- 点要**不共线**，而且尽量散开（张成一个体积，不要都在一条线或一个平面附近）；
- 每个点两条臂各碰一次，碰的是**同一个物理特征**，不是"差不多的位置"；
- 全程只读连接，不下发运动，靠手动拖动示教把末端摆过去。

**判读**

| 结果 | 含义 | 下一步 |
|---|---|---|
| 与抄来的值一致（残差小） | 镜像是对的 | 进入第 3 步，并把 `provenance` 改成实测 |
| 不一致 | 镜像是错的 | **在演示前发现了它**。用实测值替换，再重跑一次验证 |

无论哪种结果，都要把残差和采点数写进 `safety_profiles.json` 的
`left_tool_mount_calibration.evidence_id`，不要只改数值不留证据——
这个字段存在的意义就是让下一个人知道这个值是怎么来的。

相关工具：`scripts/measure_left_arm_bridge.py`、`scripts/solve_left_arm_model.py`
（左臂的 SDK↔MoveIt 对应关系已于 2026-08-04 解出并落库为
`safety_profiles.json` 的 `left_arm_model`，**这一项不需要重做**）。

---

## 3. 开闸

`mtc_ws/src/grabber_mtc_planner/config/dual_rm75_arms.yaml`：

```yaml
- arm_id: left_arm
  execution_eligible: true
  execution_block_reason: ""
```

改完需要 `colcon build --packages-select grabber_mtc_planner`。

**注意**：这个字段**从不影响规划**——文件开头就写着"两条臂永远都规划"。
它只控制"规划出来的东西允不允许执行"。所以翻它之前，
左臂的 plan-only 结果早就存在，可以先看再翻。

---

## 4. 两处写死右臂 IP 的地方

```
scripts/execute_mtc_trajectory.py:130   cfg.connections.right_arm_ip
scripts/calibrate_mtc_gripper.py:68     cfg.connections.right_arm_ip
```

两个脚本都需要加 `--arm`（其余脚本已经有了，见第 5 节），
按 `--arm` 选 `cfg.connections.right_arm_ip` 或 `left_arm_ip`。

改完记得把它们保持在 `scripts/robot_code_drift.py` 的 `TRACKED` 列表里
（它们已经在），并用 `python scripts/robot_code_drift.py --push` 同步到机器人。

---

## 5. 已经就绪、不用动的部分

| 项 | 状态 |
|---|---|
| 规划组 | `dual_rm75_arms.yaml` 两条臂配全，`left_arm` / `l_link7` / `touch_links` 齐备 |
| 规划通路 | `ros/plan_once.py` 按规划组推导关节名和 IK 连杆，两臂共用同一条路径 |
| 电子围栏 | 已用左臂自己的基座系表达；左臂 TCP 点会换算进右控制器基座系后跑同一套检查 |
| 进程隔离 | `arm_worker.py` 给每条臂独立进程。**这不是优化**：SDK 的 `Algo` 是进程全局的，安装角、工具系、关节限位设在库上而不是句柄上，同一进程里开第二个会话会静默覆盖第一个的运动学 |
| 夹爪控制 | `--arm` 参数本来就有（`shelf_row_template_tool.py`、`gripper_grasp_test.py` 默认值甚至就是 `left_arm`）；`arm_worker` 白名单含 `gripper_state` / `open_gripper` / `close_gripper` / `close_empty_gripper` |
| 左臂示教位形 | `grasp_start_left_joints_deg` 已于 2026-08-04 由操作员手动摆位后落库，落库前逐项验过：12 秒静止漂移 0.004°、最小关节余量 8.5°、距围栏走廊边缘 0.441 m、MoveIt 自碰撞通过 |
| 抓取点示教 | `outputs/row_templates.json` 的 122 个点中 **72 个带左臂参考位形**（50 个只有左臂够得到，22 个两臂都够得到） |

---

## 6. 左臂的空夹基线要单独标一次

"抓住了"这件事的判据是**闭合位置高于本轮实测的空夹基线**，不是"发过闭合指令"。
这个基线**每条臂、每轮都要自己测**，不能共用右臂的
（`validate_holding_gripper_feedback` 在没有本轮基线时会直接拒绝判定）。

`scripts/calibrate_mtc_gripper.py` 加上 `--arm` 之后，
在自由空间跑一次左臂的空夹标定即可。

---

## 7. 覆盖范围：数据已经算好了

按 `row_templates.json` 的 `eligible_arms` 统计：

```
只有左臂够得到   50 个点
只有右臂够得到   50 个点
两臂都够得到     22 个点
```

**正好一半一半，中间 22 个重叠。** 所以"一条臂负责一半"有实测依据，
中间那 22 个可以任选一臂，也可以留作冗余。

---

## 8. 建议的验证顺序

1. **触碰验证**（第 2 节）——不动闸门，先拿到数据
2. 一致 → 更新 `provenance` 与 `evidence_id`；不一致 → 用实测值替换后重验
3. **翻闸 + 重新编译**（第 3 节）
4. 两个脚本加 `--arm`（第 4 节），推送同步
5. **左臂空夹标定**（第 6 节）
6. **左臂 plan-only**：挑一个 `eligible_arms` 只含 `left_arm` 的模板点，
   跑一次只规划，确认能解出且审计通过——**此时还没有任何运动**
7. **低速真机抓取一次**，人守急停

第 6 步之前不需要机械臂运动；第 2 步需要人手拖动但不下发指令。
真正的运动只在第 7 步。

---

## 9. 已知的、与左臂无关但会干扰调试的两件事

- **头部 RealSense（序列号 153122071777）会从设备枚举中消失**，
  2026-08-07 出现三次、整机重启无效，报
  `未找到 RealSense 153122071777；当前设备: ['405622073249','335522072194']`。
  它会让任何依赖头部相机的步骤随机失败。左臂调试若卡在采集，先排除它。
- **RGB-D 障碍体素比实物胖**，详见 `docs/rgbd_voxel_inflation.md`。
  左臂的抓取规划会遇到和右臂一样的问题，不是左臂特有的。
