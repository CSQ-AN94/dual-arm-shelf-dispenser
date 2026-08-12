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

## B. 复现昨天的抓取结果

昨天（2026-08-11）四次三成：外星人上层 0.26、柠檬茶上层 0.54、阿萨姆下层 0.56。
证据在 `outputs/runs/20260811_picks/`。

### B.1 上层抓一个

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

### B.2 下层抓一个

把 `LAYER=upper` 改成 `LAYER=lower`。它会先归位到 647、再空手降到 250 再抓。

### B.3 判成败

**看 `pick_record.json` 在不在，不要只看日志最后一行。**
（2026-08-11 有一次执行成功了但脚本被中断，横幅没打出来，被误判成失败。）

```bash
ssh rm@192.168.3.68 'ls -la /home/rm/pick_now/pick_record.json 2>/dev/null && echo "抓取成功" || echo "没抓到"'
```

### B.4 已知的位置限制

**下层行位 0.32~0.34 抓不了**（维他命水、美年达都栽在这），
原因是关节直线路线在那个位置会撞，只剩随机采样，出来的路 900~3500°，
切成 15° 一段要 61~239 条指令，而控制器队列上限 29。

操作者实测：**那个位置人手能抓，但要斜着进，不能笔直进去** ——
所以是逼近方向的问题，不是够不到。**在修好之前，把瓶子放在 0.45~0.55。**

---

## C. 跨层搜索（新功能，2026-08-12 首次真机验证通过）

不知道东西在哪一层时，让它自己找：

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && timeout 420 /home/rm/miniconda3/envs/tube_vision/bin/python3 scripts/find_product_shelf_layer.py --target-product P06 2>&1 | tail -6'
```

它遍历 `[647, 250]`，空载升降到每层用头部相机找，找到就打印
`SHELF_LAYER_LIFT_MM=<高度>` 并退出 0；每层都没有退出 3（补货，不是故障）。

**跑之前确认双臂都在收拢位**（A.3）——它会移动升降，而升降移动时手臂不能挂在货架里。

**已知 bug（未修）**：判"这层有没有"只看检测命中数是不是 0，
所以**一帧误检就会把"这层没有"变成"故障"并中止整个搜索**（实测 `1/7` 中止）。
真要用，先看它报的是 `0/7`（正常换层）还是 `1/7`（会中止）。

**它现在还没接进抓取主线** —— 主线里仍然是手写 `LAYER=upper|lower`。
接法见 [`GRASP_MAINLINE.md`](GRASP_MAINLINE.md) §2.1。

---

## D. 整套流程：现在到哪了，接下来按什么顺序

### D.1 现状一张表

| 环节 | 状态 | 证据 |
|---|---|---|
| 归位（双臂+升降原子门禁） | ✅ 稳定 | 每次运行 |
| **抓取（MTC 主线）** | ✅ **4 次 3 成** | `outputs/runs/20260811_picks/` |
| 跨层搜索 | ✅ 首次真机通过 | 2026-08-12 |
| 收臂 | ✅ 稳定 | 同上 |
| 库存数据库 | ⬜ 写好，**零调用者** | `inventory.py` |
| 放回下层货架 | ❌ **自主放置 0 成功** | 最后一个卡点已修，未验证 |
| 侧桌投放（转身放桌上） | ❌ **代码通了，零真机** | 见 §D.3 |

### D.2 推荐顺序（每步都能独立验收）

**第 1 步：修下层内侧行位抓不了**（B.4）
逼近方向要能斜着进。改完用 0.32 那个位置验收——现在必失败，改完该成功。
不占机器人的部分：MTC 的 `source_approach_direction` 是怎么定的。

**第 2 步：把跨层搜索接进主线**（`LAYER=auto`）
验收：`PRODUCT_CODE=P06 LAYER=auto` 能自己找到层并抓到。
一个接口毛刺：搜索高度表读的是 `side_table_template`，货架抓取不该依赖投放 profile。

**第 3 步：接库存**（三处，见 `GRASP_MAINLINE.md` §2.2）
验收：扫一层之后 `inventory.json` 里有那层每个槽位的内容；
下次要同一个东西时 `find()` 命中、直接去那层。

**第 4 步：放回下层货架**（比侧桌便宜得多的出货路径）
`PICK_ONLY=0` 跑完整循环。最后一个已知卡点（悬空碎块）已修但没上机验证过。

**第 5 步：侧桌投放**，分三段验，每段之间停下来看：
1. **空手、不带瓶**，只跑旋转：转过去 → 停 → 转回来
2. 空手转过去后，只跑桌面观测干跑，看 ROI 里能不能找出候选
3. 前两段都过了，再带瓶跑全程

### D.3 侧桌投放必须知道的三件事

1. **它建在旧管线上**（`run_task.sh` → `run_pick_place_task.py` → 右腕观察位那一套），
   不是主线的 MTC 抓取。2026-08-12 四次失败里有两次直接来自这个事实。
   **出路是让它改用主线的抓取，不是继续在旧管线上补窟窿。**
2. **底盘旋转是整机扫掠**，是这个项目里后果最大的动作，从未真机执行过。
3. **`rotation_sweep.clearance_m = 0.30` 是手输的、没量过的**，
   `_assert_sweep` 分辨不了实测值和手输值 —— 真正的保护是人守急停。

---

## E. 出问题先看这三条

### E.1 头部相机掉线

报 `未找到 RealSense 153122071777`。2026-08-08 出现三次，**整机重启无效**。
先换 USB 口。跑之前可以先确认：

```bash
ssh rm@192.168.3.68 'cd /home/rm/dual-arm-shelf-dispenser && /home/rm/miniconda3/envs/tube_vision/bin/python3 -c "
import pyrealsense2 as rs
for d in rs.context().query_devices(): print(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
"'
```

看到 `153122071777` 才是好的。

### E.2 关节报 0xF000（通信丢帧）

**拖动示教之后很常见**（按住绿色拖动按钮会把这个标志锁住）。
预检会自动尝试清除，一般不用管；拦住了就重跑一次。

### E.3 抓取"未解出"

看 `p*.json` 里的 `complete_solution_count_by_arm`：

- **有完整解但全被否** → 执行审计拒了，看 `pp*.log` 里 `rejecting ... solution:` 那行哪几项是 `false`
- **零完整解** → 规划没解出来，看 `earliest_failure_stage_by_arm`

参考值：成功的那几轮也只是 20 条解里活下来 2~6 条，**审计否掉九成是常态**。

---

## F. 相关文档

| 文档 | 什么时候看 |
|---|---|
| [`GRASP_MAINLINE.md`](GRASP_MAINLINE.md) | **改抓取代码之前必读** —— 什么是主线、什么不是 |
| [`why_the_place_failed_20260812.md`](why_the_place_failed_20260812.md) | 要跟别人解释放置为什么没成 |
| [`side_table_delivery_state_20260812.md`](side_table_delivery_state_20260812.md) | 做侧桌投放之前，哪些数是实测的、哪些是编的 |
| [`left_arm_bringup.md`](left_arm_bringup.md) | 要动左臂 |
| [`rgbd_voxel_inflation.md`](rgbd_voxel_inflation.md) | 体素虚胖 / 点云噪点 |
