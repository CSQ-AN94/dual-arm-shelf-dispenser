# bottle_grasp — 双臂货架售货抓取机器人（技术验证中）

写于 2026-07-20，`dual-arm-sdk` 分支。这是仓库两条技术路线之一的实际代码和
运行说明（另一条是根目录 `direct_grab.py` 的单臂+UGV架构，见根目录
[README.md](../README.md)）。这份文档是"具体怎么跑、怎么改"的实用入口；
更详细的历史背景、踩过的坑、逐功能点验证状态在
[docs/handoff/](../docs/handoff/)，不在这里重复。

## 这是什么

双臂 Realman（RM75-BI）+ 头部/双腕三个 RealSense 深度相机的固定式平台。
`bottle_grasp/` 是一套完全独立的新 SDK 控制栈（不碰旧的
`controllers/arm_controller.py`）：真实 TCP 连接、力控夹爪、MoveIt2 桥接
做碰撞规划、外加一层跟 MoveIt 完全独立的电子围栏（笛卡尔空间硬校验）做
双重把关。核心设计原则："抓取状态机跟环境无关，静态geometry和允许区域
全部放在 `safety_profiles.json` 里，换场景只换配置，不改状态机代码"
（细节见 [SAFETY_PROFILES.md](SAFETY_PROFILES.md)）。

**现在的开发目标不是"做出一台能卖的售货机"，是验证这套抓取/安全规划/
避障架构本身**。`table_demo` 曾有桌面单瓶抓放历史记录，但右夹爪工具安装
变换现已被审计为未证实；因此所有随仓库提交的右臂 profile 都是离线候选，不能执行。
货架多格位、真实出货（送到取货口）、按商品视觉识别同样没有可用的真机授权，
详见下面“现状”一节。

## 快速开始

### 0. 本地跑逻辑测试（不需要机器人，随时能跑）

```bash
# Mac 上（或任何装了 pytest/numpy/scipy 的环境）
python -m pytest test/bottle_grasp/
```

只证明"逻辑没被改坏"，对上机成功率没有预测力——这是仓库测试文档反复
强调的一句话，别当成"验证过"。

### 1. 板房体检（上任务前必跑，零机械臂运动）

```bash
scripts/run_bottle_grasp_site_check.sh
```

时钟 → 头部舵机基准 → MoveIt碰撞三态探针 → 相机/SDK/MoveIt栈 → 右臂健康 →
夹爪反馈 → 头部深度流 → 真实YOLO定位 → 桌面拟合+场景预算 → 完整规划彩排 →
右腕相机流。全绿输出 GO（退出码0），任何一项失败 NO-GO（退出码2）。

### 2. 桌面 demo（`table_demo`，历史记录；当前执行门禁关闭）

```bash
# 右臂因安全检查停在距目标约 8.5 cm 的预抓取悬停位；只续跑最后抓放段
scripts/run_bottle_grasp.sh from-pregrasp

# 右臂已经在腕部观察位
scripts/run_bottle_grasp.sh from-observation

# 从头部定位开始完整跑一遍（含返回初始姿态）
scripts/run_bottle_grasp.sh from-start
```

这些命令保留为历史/离线接口参考；当前 profile 会在任何硬件初始化前 fail-closed。
可覆盖的环境变量均会在同步/SSH 前校验；默认值保持既有完整 task-mode 路径：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ROBOT_HOST` | `rm@192.168.3.68` | SSH 目标 |
| `REMOTE_DIR` | `/home/rm/Grabber` | 机器人上的仓库路径 |
| `REMOTE_PY` | `/home/rm/miniconda3/envs/tube_vision/bin/python` | 机器人上的 Python |
| `SAFETY_PROFILE` | `table_demo` | 电子围栏 profile 名 |
| `PORT` | `8879` | 机器人本地 dashboard 端口 |
| `DISPENSE` | `0` | `1` 时加 `--dispense`（见下面货架场景） |
| `TARGET_PRODUCT` | 空 | 非空时加 `--target-product <值>`（见下面货架场景） |
| `COMMISSIONING_SPEED` | 未设置 | `1-100` 的全运动速度上限；不设置即保留既有速度默认值 |
| `BOTTLE_GRASP_TRAJECTORY_MODE` | `continuous` | `continuous` 为正常控制点交融；`blocking` 为逐点阻塞回退 |
| `VISUAL_MODE` | 由 `VISUAL_SERVO` 推导 | `off` 保持原路径；`shadow` 只记录建议修正；`active` 执行受限修正 |
| `VISUAL_SERVO` | `0` | 兼容旧命令：`0=off`，`1=active` |
| `STOP_AFTER_OBSERVATION` | `0` | `1` 时仅到观察位并完成定位后结束，不下发闭夹；仅支持 `from-start`/`from-observation` |
| `CONFIRM_BEFORE_GRASP` | `0` | `1` 时定位后在同一进程等待终端 Enter 再抓取；与 `STOP_AFTER_OBSERVATION=1` 互斥 |

两个任务都会在接管 SDK 前停掉 `atom`/`zhixing_ctrl.py` 遥操，**任务结束不
自动恢复**，需要另外手动跑官方 `upstart_all.sh`（见
[demo_operation_guide.md](../docs/demo_operation_guide.md) 或
`docs/bottle_grasp_demo_runbook.md`）。

### 3. 货架场景（2026-07-20 新增代码，还没有现场验证）

先量货架（零机械臂运动，只起头部相机）：

```bash
scripts/run_measure_shelf_geometry.sh 0.0 0.55 -0.10 \
  shelf_bottom,shelf_back,shelf_left_panel,shelf_right_panel
```

打印每个面的草稿坐标 + `keepout_boxes` JSON 片段，**人工核对修正后**手动
合并进 `bottle_grasp/safety_profiles.json` 的 `shelf_template`（id 必须是
`shelf_model.FACE_SPECS` 认的名字：`shelf_bottom`/`shelf_top`/`shelf_back`/
`shelf_left_panel`/`shelf_right_panel`），同时现场量出货口坐标填
`output_joints_deg`。全部填完、且 `enabled: true` 之后，先运行 site-check
规划彩排：

```bash
SAFETY_PROFILE=shelf_template \
scripts/run_bottle_grasp_site_check.sh
```

task-mode 只接受真实 `--execute` 交易，不能把 `--plan-only` 伪装成分级任务。
上机前的规划彩排入口是 `scripts/run_bottle_grasp_site_check.sh`：它用真实场景和
规划链路生成证据，但不执行机械臂运动。

**P0 集成门禁：**当前 `table_demo`、`shelf_template` 和无环境避障 profile
均为 `verified_for_execution=false`。右夹爪相对 MoveIt `r_link7`、控制器法兰和
物理 TCP 的完整安装旋转尚无可复现实测证据；不得用原来的纯 `+Z` 偏移猜测。
恢复任何右臂执行前，必须填入并复核 `tool_mount_calibration` 的两段 4×4 刚体变换、
证据编号、时间和位置/姿态残差。仅可先做不发运动的 site-check/只读核验。

`TARGET_PRODUCT` 要求视觉模型能按商品类别识别（不是通用"瓶子"），这部分
现在还没有——见下一节。

### 4. 商品识别模型训练（如果要用 `--target-product`）

完整流程（采集 → LabelImg人工打框 → 整理数据集 → 训练 → 验证 → 接回
`config.yaml`）见 [product_yolo_training.md](../docs/handoff/product_yolo_training.md)。
命令速览：

```bash
# 机器人上，按商品分类采集（每个商品单独跑一次）
python scripts/collect_product_images.py --label coke_bottle --camera head

# 人工用 LabelImg/CVAT 打框（YOLO格式），存进
# intelligence/data/raw/<label>/ 同名 .txt

# Mac/训练机上，整理成训练集
python scripts/prepare_yolo_dataset.py \
  --raw-root intelligence/data/raw --output-root intelligence/data/product_yolo

# 训练机（推荐 RTX 4060 这类真实CUDA GPU）上微调
python scripts/train_product_yolo.py \
  --data intelligence/data/product_yolo/data.yaml --device cuda:0

# 验证类别/画框
python scripts/verify_yolo_model.py <best.pt路径> --image 一张真实图.jpg
```

## 现状（逐场景，别笼统说"能跑"）

| 场景 | 状态 |
|---|---|
| `table_demo`（桌面单瓶，放回原位） | 2026-07-19 有3次历史抓放记录（无避障入口）；P0 审计发现工具安装旋转未证实，当前执行 **NO-GO**，见 [bottle_grasp_status.md](../docs/handoff/bottle_grasp_status.md) |
| 货架多面电子围栏自适应 | 代码+单测（`shelf_model.py`，24个测试）已完成，**零真机验证** |
| 真实出货 `_deliver_to_output` | 代码+单测已完成，`output_joints_deg`/`output_point_base` 在 `shelf_template` 里还是占位/空值，**零真机验证** |
| 按商品视觉识别 `--target-product` | 参数链路已打通，**没有训练过任何多类别模型**，现在的 `8_17.pt` 只认通用瓶子 |
| 现场测量工具 | `measure_shelf_geometry.py` 逻辑+单测完成，**没有在真实货架上跑过** |

## 架构速览

```
bottle_grasp/
  core.py            共享数据类型（DemoParams参数表、Localization、SafetyAbort）
  robot.py            RobotSession：真实TCP/SDK连接、力控夹爪、正逆解
  planner.py / safe_planner.py   MoveIt桥接 + 候选排序/围栏反馈/双重后验复核
  safety.py / safety_profiles.json   电子围栏：FenceBox/SafetyProfile
  table_model.py       桌面单水平面每轮自适应（table_demo专用，已验证）
  shelf_model.py       货架五个面每轮自适应（泛化自table_model.py，未验证）
  shelf_survey.py       现场一次性测量，产出草稿box
  perception.py         YOLO检测，target_classes按商品筛选
  demo.py               BottleDemo状态机主体
  task.py               三个明确入口(from-pregrasp/from-observation/from-start) + DeliverMode(放回/出货)

scripts/
  run_bottle_grasp_site_check.sh   板房体检
  run_bottle_grasp.sh              唯一公开实机任务入口
  measure_shelf_geometry.py + run_measure_shelf_geometry.sh   现场货架测量
  collect_product_images.py / prepare_yolo_dataset.py /
  train_product_yolo.py / verify_yolo_model.py                商品识别模型训练流程

test/bottle_grasp/   280个测试，纯逻辑+mock，不连真机/ROS
```

## 深入阅读

- [docs/bottle_grasp_demo_runbook.md](../docs/bottle_grasp_demo_runbook.md) — 现场怎么跑、验收标准，唯一权威
- [docs/handoff/bottle_grasp_status.md](../docs/handoff/bottle_grasp_status.md) — 完整现状、逐功能点验证表、下一步优先级
- [docs/handoff/bottle_grasp_known_risks.md](../docs/handoff/bottle_grasp_known_risks.md) — 已踩过的坑 + 现在还没解决的风险点，改代码前先看
- [docs/handoff/obstacle_avoidance.md](../docs/handoff/obstacle_avoidance.md) — MoveIt/电子围栏/自动重规划的领域背景
- [SAFETY_PROFILES.md](SAFETY_PROFILES.md) — 电子围栏 profile 的配置格式和货架/出货点适配说明
- [docs/handoff/product_yolo_training.md](../docs/handoff/product_yolo_training.md) — 商品识别模型训练完整流程
- [docs/抓水瓶教程/](../docs/抓水瓶教程/) — 零基础教学五册（抓取点估计/路径规划/机械臂控制/避障/奇异点）
