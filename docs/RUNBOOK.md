# 运行手册

**这是唯一的运行手册。** 命令都是完整的，在 Mac 上任何目录粘贴就能跑。
机器人 `rm@192.168.3.68`，仓库 `/home/rm/dual-arm-shelf-dispenser`，
机器人上的 Python 是 `/home/rm/miniconda3/envs/tube_vision/bin/python3`。

**硬约束：全程不开遥操。** 不要跑 `upstart_all.sh`，不要恢复 `atom` / `zhixing_ctrl.py`。

---

## A. 开机必做（三条，按顺序）

### A.1 推代码

```bash
cd "/Users/siqi.cai/Embodied AI/dual-arm-shelf-dispenser" && /Users/siqi.cai/miniconda3/envs/robo/bin/python scripts/robot_code_drift.py --push
```

看到 `✓ N 个文件已同步并复验一致` 才算过。**新加的脚本要自己加进 `TRACKED`**，
否则它不会被推上去，而且照样打印"已同步"。

### A.2 编译规划器

C++ 改过就必须编译，不编译不生效（约 50 秒）：

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser/mtc_ws && source /opt/ros/humble/setup.bash && source /home/rm/ros2_ws/install/setup.bash && colcon build --packages-select grabber_mtc_planner 2>&1 | tail -3'
```

### A.3 看机器人在什么状态

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && timeout 150 /home/rm/miniconda3/envs/tube_vision/bin/python3 -c "
import sys; sys.path.insert(0,\".\")
from shelf_dispenser.core import DemoParams
from shelf_dispenser.live_arm import open_live_arm
from shelf_dispenser.safety import load_safety_profile
from shelf_dispenser.mobile_body import LiftSocketAdapter
from utils.config import load_config
cfg=load_config(\"config.yaml\")
prof=load_safety_profile(\"shelf_dispenser/safety_profiles.json\",\"shelf_template\",require_verified=False)
print(\"升降\", LiftSocketAdapter(cfg.connections.left_arm_ip, cfg.connections.arm_port).state().height_mm, \"mm\")
for arm in (\"right_arm\",\"left_arm\"):
    r,v=open_live_arm(cfg,DemoParams(),prof,arm,take_control=False)
    q=r.joints_deg(); g=r.gripper_state()
    t=prof.grasp_start_right_joints_deg if arm==\"right_arm\" else prof.grasp_start_left_joints_deg
    print(\"%s 离归位 %.2f deg  夹爪 state=%s pos=%s\"%(arm,max(abs(a-b) for a,b in zip(q,t)),g[\"dof_state\"][0],g[\"pos\"][0]))
    r.close()
" 2>&1 | tail -4'
```

**怎么读**：`离归位` < 1° 就是在收拢位。夹爪 `state=3` 是夹住了东西、`state=2` 是没夹住，
`pos` 越大越张开（全开约 900，0 是闭合）。**两个都读成 0 是夹爪掉线**，
只有给工具口断电能救，而且**手里有东西时不能断电**。

---

## B. 零件命令（随时可单独跑，互不依赖）

日常最常用的就是这一节。**每条都能单独跑**，不需要先跑别的。

### B.1 右臂归位

**不动升降**（快，1 秒启动）：

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && timeout 300 /home/rm/miniconda3/envs/tube_vision/bin/python3 scripts/normalize_right_arm.py --skip-lift --speed 20 --execute 2>&1 | tail -2'
```

**连升降一起归到 647**：

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && timeout 300 /home/rm/miniconda3/envs/tube_vision/bin/python3 scripts/normalize_right_arm.py --speed 20 --execute 2>&1 | tail -2'
```

去掉 `--execute` 就是干跑，只报偏差不动。

### B.2 左臂归位

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && timeout 300 /home/rm/miniconda3/envs/tube_vision/bin/python3 scripts/normalize_left_arm.py --execute 2>&1 | tail -2'
```

### B.3 夹爪开合（两条臂都支持）

**张开**：

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && timeout 200 /home/rm/miniconda3/envs/tube_vision/bin/python3 scripts/shelf_row_template_tool.py open-gripper --arm right_arm 2>&1 | tail -2'
```

**收拢**（只合爪，不判定有没有抓到，不动关节）：

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && timeout 200 /home/rm/miniconda3/envs/tube_vision/bin/python3 scripts/shelf_row_template_tool.py close-gripper --arm right_arm 2>&1 | tail -2'
```

把 `right_arm` 换成 `left_arm` 就是左臂。

**完整抓取测试**（张开 → 合拢 → 判定抓没抓到）：

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && timeout 300 /home/rm/miniconda3/envs/tube_vision/bin/python3 scripts/gripper_grasp_test.py --arm right_arm 2>&1 | tail -6'
```

### B.4 升降

**升到 647**：用 B.1 的第二条（归位时顺带升）。

**降到 250**（直接驱动，不需要 pick 记录）：

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && timeout 200 /home/rm/miniconda3/envs/tube_vision/bin/python3 -c "
import sys; sys.path.insert(0,\".\")
from shelf_dispenser.mobile_body import LiftSocketAdapter
from utils.config import load_config
cfg=load_config(\"config.yaml\")
lift=LiftSocketAdapter(cfg.connections.left_arm_ip, cfg.connections.arm_port)
print(\"当前\", lift.state())
print(\"结果\", lift.move_to(250, speed=30))
" 2>&1 | tail -3'
```

改 `250` 就是改目标高度。**速度上限 30，写更大会被拒**。

⚠️ **降之前必须确认右臂在收拢位**（先跑 B.1 第一条），否则臂可能挂在货架里被拖下去。

### B.5 头部基准角（检查 / 复位）

**整条抓取主线的定位只靠这个固定头部相机**，头一动这一轮的三维点就全错，
所以采集前会校验、不在基准角直接拒绝，而且**不会自动去转它**。

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && timeout 120 /home/rm/miniconda3/envs/tube_vision/bin/python3 scripts/head_position_lock.py check 2>&1 | tail -5'
```

不在基准角时复位（绝对位置命令 + 闭环复核）：

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && timeout 200 /home/rm/miniconda3/envs/tube_vision/bin/python3 scripts/head_position_lock.py restore 2>&1 | tail -6'
```

**基准值**（`shelf_dispenser/head_lock.py`，2026-07-08 标定实测）：

| | 值 | 含义 |
|---|---|---|
| `angle1` | **398** | 俯仰，最低位 |
| `angle2` | **516** | 偏航，居中 |
| 容差 | ±5 | 舵机反馈本身有几个单位抖动 |

⚠️ 两条光看数字不会知道的：

- 厂商服务把 `angle1` 的**命令**下限卡在 400，而**反馈**在机械限位附近是 398±几
- **厂商方向键固定 50 步长，到不了 `angle2=516`** —— 不要用方向键去凑，只能用上面的 `restore`
- 这台机器的 **pitch 舵机（ID=1）有已知问题**：相机上下动不了、只能左右动，
  所以 `angle1` 实际就停在机械限位上

### B.6 回放一条示教轨迹

**先干跑**（检查围栏、路点数、起点对不对得上，不动）：

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && timeout 200 /home/rm/miniconda3/envs/tube_vision/bin/python3 scripts/replay_demonstrated_trajectory.py outputs/demonstrated_trajectories/right_lower_place_E_mid_20260807.json --speed 10 2>&1 | tail -8'
```

干跑通过后加 `--execute` 执行；加 `--reverse` 原路退回；
加 `--stop-at-fraction 0.29` 只回放前一段。

**干跑会检查三件事**：录制时围栏违规数、简化后路点数（上限 29）、
**实机当前位形离示教起点差多少**（超过 2° 就拒绝）。三条都过才敢 `--execute`。

⚠️ **这个脚本只查电子围栏，不查碰撞场景**。货架上瓶子换过位置之后，
一条旧轨迹可能会扫到新摆的瓶子。**每次回放前先看一眼货架实物。**

---

## C. 复现昨天的抓取结果

昨天（2026-08-11）四次三成：外星人上层 0.26、柠檬茶上层 0.54、阿萨姆下层 0.56。
证据在 `outputs/runs/20260811_picks/`。

### C.1 上层抓一个

把瓶子放上层，**行位靠右**（右臂舒服区 0.45~0.55），然后：

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && PRODUCT_CODE=P01 LAYER=upper PICK_ONLY=1 CYCLE_OUT=/home/rm/pick_now setsid nohup bash scripts/run_cross_layer_cycle.sh > /home/rm/pick_now.log 2>&1 < /dev/null & sleep 3; echo started'
```

看进度：

```bash
ssh rm@192.168.3.68 'tail -f /home/rm/pick_now.log'
```

`PRODUCT_CODE` 对照：`P01` 外星人、`P02` 维他命水、`P03` 阿萨姆、
`P04` 维他柠檬茶、`P05` 百事、`P06` 美年达。

### C.2 下层抓一个

把 `LAYER=upper` 改成 `LAYER=lower`。它会先归位到 647、再空手降到 250 再抓。

### C.3 判成败

**看 `pick_record.json` 在不在，不要只看日志最后一行。**
（2026-08-11 有一次执行成功了但脚本被中断，横幅没打出来，被误判成失败。）

```bash
ssh rm@192.168.3.68 'ls -la /home/rm/pick_now/pick_record.json 2>/dev/null && echo "抓取成功" || echo "没抓到"'
```

### C.4 已知的位置限制

**下层行位 0.32~0.34 抓不了**（维他命水、美年达都栽在这），
原因是关节直线路线在那个位置会撞，只剩随机采样，出来的路 900~3500°，
切成 15° 一段要 61~239 条指令，而控制器队列上限 29。

操作者实测：**那个位置人手能抓，但要斜着进，不能笔直进去。**

而把 122 个示教点的逼近方向全算了一遍（2026-08-12）：

| 层/行位 | 逼近方向 | 离"正对水平"差 |
|---|---|---|
| lower 0.34 | `[0.012, -0.996, -0.087]` | 5.1° |
| lower 0.56 | `[-0.050, -0.998, -0.048]` | 4.0° |
| upper 0.60 | `[-0.010, -1.000, -0.027]` | 1.6° |

**整个示教库里没有一个斜向抓取，最大偏离正对才 11°。** 抓取姿态是从这些点插值出来的，
所以代码里根本没有能表达"斜着进"的东西——**这需要重新示教那几个内侧点**
（`record_demonstrated_grasp.py` + `shelf_row_template_tool.py`），不是改参数。

### 下层 0.32~0.40：用示教回放，不要等规划

2026-08-13 同一个瓶子、同一个位置、同一时刻做的对照：

| | 结果 |
|---|---|
| 自动规划（美年达，行位 0.40） | ❌ **22 条完整解全被否**，自由段绕 917~3370° |
| 回放示教路径 | ✅ **抓到了**（`state=3 pos=340 current=98`） |

失败的三个数据点是 **0.32 / 0.34 / 0.40**，成功过的是 **0.56**。
失败全部发生在自由段 `connect_to_source_pregrasp`（直线撞 → 只剩采样 → 绕几千度
→ 切成 15° 一段要 61~225 条指令，队列上限 29），**跟抓取姿态无关**。

**这一带现在的做法**（正反两向都真机验证过）：

```bash
# 1. 右臂归位到收拢位（回放的起点），升降 250，右爪张开
# 2. 正向回放到抓取位
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && timeout 500 /home/rm/miniconda3/envs/tube_vision/bin/python3 scripts/replay_demonstrated_trajectory.py outputs/demonstrated_trajectories/pick_B3_right_20260813.json --speed 10 --execute 2>&1 | tail -6'
# 3. 闭合夹爪（B.3），确认 state=3 且 pos 远高于空夹基线
# 4. 要放回去就张开夹爪，再加 --reverse 原路退回
```

⚠️ **回放只查电子围栏、不查实时碰撞场景**。示教之后货架上别的瓶子挪过位置，
这条旧路径可能会扫到它们 —— **每次回放前先看一眼货架实物**。

**在这一段的规划修好之前，把瓶子放在 0.45~0.55 是最省事的做法。**

---

## D. 跨层搜索（新功能，2026-08-12 首次真机验证通过）

不知道东西在哪一层时，让它自己找：

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && timeout 420 /home/rm/miniconda3/envs/tube_vision/bin/python3 scripts/find_product_shelf_layer.py --target-product P06 2>&1 | tail -6'
```

它遍历 `[647, 250]`，空载升降到每层用头部相机找，找到就打印
`SHELF_LAYER_LIFT_MM=<高度>` 并退出 0；每层都没有退出 3（补货，不是故障）。

**跑之前确认双臂都在收拢位**（A.3）——它会移动升降，而升降移动时手臂不能挂在货架里。

**"几中几"已修**（2026-08-12）：`1~2` 个可用点算"这层没有"、继续下一层；
`0` 个但检测有命中算故障（深度坏了）；`3~6` 个仍算故障（足够多的帧认同却定不下来）。
分界 3 来自共识判据自己的下限。

**主线已经内建了逐层扫描**：`PICK_ONLY=1` 时 `LAYER` 默认就是 `auto`，
`run_cross_layer_cycle.sh` 自己按 upper → lower 逐层空载升降 + 头部检测，
不调这个独立脚本、也不依赖投放 profile 的高度表。上面这条命令用于
**单独验证搜索本身**（不抓取），或者需要先知道东西在哪一层再决定怎么做。

---

## E. 整套流程：现在到哪了，接下来按什么顺序

### E.1 现状一张表

| 环节 | 状态 | 证据 |
|---|---|---|
| 归位（双臂+升降原子门禁） | ✅ 稳定 | 每次运行 |
| **抓取（MTC 主线）** | ✅ **4 次 3 成** | `outputs/runs/20260811_picks/` |
| 跨层搜索 | ✅ 首次真机通过 | 2026-08-12 |
| 收臂 | ✅ 稳定 | 同上 |
| 库存数据库 | ⬜ 写好，**零调用者** | `inventory.py` |
| 放回下层货架 | ❌ **自主放置 0 成功** | 最后一个卡点已修，未验证 |
| 侧桌投放（转身放桌上） | ❌ **代码通了，零真机** | 见 §E.3 |

### E.2 推荐顺序（每步都能独立验收）

**第 1 步：示教下层内侧行位的斜向抓取**（C.4）
不是改代码——示教库里一个斜向抓取都没有。要在 0.30~0.36 那一带按你手动成功的角度
录几个点，进行模板。验收：现在 0.32 必失败，示教之后该成功。

**第 2 步：验证主线自带的 `LAYER=auto`**（已接线，未真机跑过整条）
验收：`PRODUCT_CODE=P06 PICK_ONLY=1` 不指定 LAYER，能自己找到层并抓到。

**第 3 步：接库存**（三处，见 `GRASP_MAINLINE.md` §2.2）
验收：扫一层之后 `inventory.json` 里有那层每个槽位的内容；
下次要同一个东西时 `find()` 命中、直接去那层。

**第 4 步：放回下层货架**（比侧桌便宜得多的出货路径）
`PICK_ONLY=0` 跑完整循环。最后一个已知卡点（悬空碎块）已修但没上机验证过。

**第 5 步：侧桌投放**，分三段验，每段之间停下来看：
1. **空手、不带瓶**，只跑旋转：转过去 → 停 → 转回来
2. 空手转过去后，只跑桌面观测干跑，看 ROI 里能不能找出候选
3. 前两段都过了，再带瓶跑全程

### E.3 侧桌投放必须知道的三件事

1. **旧管线已于 2026-08-12 停用**，`run_task.sh` 直接运行会拒绝并指向主线。
   迁移方案（哪些能力已经在库里、只有落桌伺服要搬）见
   [`GRASP_MAINLINE.md`](GRASP_MAINLINE.md) §3.1。
2. **底盘旋转是整机扫掠**，是这个项目里后果最大的动作，从未真机执行过。
3. **`rotation_sweep.clearance_m = 0.30` 是手输的、没量过的**，
   `_assert_sweep` 分辨不了实测值和手输值 —— 真正的保护是人守急停。

---

## F. 出问题先看这三条

### F.1 头部相机掉线

报 `未找到 RealSense 153122071777`。2026-08-08 出现三次，**整机重启无效**。
先换 USB 口。跑之前可以先确认：

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && /home/rm/miniconda3/envs/tube_vision/bin/python3 -c "
import pyrealsense2 as rs
for d in rs.context().query_devices(): print(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
"'
```

看到 `153122071777` 才是好的。

### F.2 关节报 0xF000（通信丢帧）

**拖动示教之后很常见**（按住绿色拖动按钮会把这个标志锁住）。
预检会自动尝试清除，一般不用管；拦住了就重跑一次。

### F.3 抓取"未解出"

看 `p*.json` 里的 `complete_solution_count_by_arm`：

- **有完整解但全被否** → 执行审计拒了，看 `pp*.log` 里 `rejecting ... solution:` 那行哪几项是 `false`
- **零完整解** → 规划没解出来，看 `earliest_failure_stage_by_arm`

参考值：成功的那几轮也只是 20 条解里活下来 2~6 条，**审计否掉九成是常态**。

---

## G. 相关文档

| 文档 | 什么时候看 |
|---|---|
| [`GRASP_MAINLINE.md`](GRASP_MAINLINE.md) | **改抓取代码之前必读** —— 什么是主线、什么不是 |
| [`why_the_place_failed_20260812.md`](why_the_place_failed_20260812.md) | 要跟别人解释放置为什么没成 |
| [`side_table_delivery_state_20260812.md`](side_table_delivery_state_20260812.md) | 做侧桌投放之前，哪些数是实测的、哪些是编的 |
| [`left_arm_bringup.md`](left_arm_bringup.md) | 要动左臂 |
| [`rgbd_voxel_inflation.md`](rgbd_voxel_inflation.md) | 体素虚胖 / 点云噪点 |
