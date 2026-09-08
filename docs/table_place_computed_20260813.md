# 2026-08-13：桌面放置第一次跑通（算出来的，不是回放的）

一句话：**瓶子放上桌了，落点是解算的；但有一个数是人喂的，而且只成功过一次。**

---

## 1. 成了什么

21:04，外星人电解质水（P01）被放到右侧桌面上，夹爪张开，手臂退开。

```
桌面放置下降观测  49.0 → 38.3 → 27.6 → 17.2 mm   四步，每步过全链碰撞复核
桌面放置完成      偏移 120.0 mm，瓶底停在桌面上方 6.0 mm
松开夹爪          ✓
三维确认          head 3/3 共识，散布 1.0 mm
```

证据：`/home/rm/pick_p01_20260813_173304/placement_computed/20260813_210327_920863_828bcabf/`
（`output_place_servo.json` 有完整下降记录）。

**入口**：`scripts/place_held_bottle_only.py`（本次从机器人上收进仓库并参数化）。

```bash
python3 scripts/place_held_bottle_only.py \
  --pick-record   <pick_record.json> \
  --demonstration <taught_place_route.json> \
  --bottle-bottom-below-tcp-m 0.12 \
  --output-dir    <dir>
```

先加 `--dry-run`：跑到"选出落点"为止，不下发任何运动。

---

## 2. 哪些是机器人自己算的，哪些不是

| 量 | 来源 |
|---|---|
| 桌面高度与倾斜 | 实时 RGB-D 三帧拟合**斜面**，帧间一致 0.9 mm |
| 空位候选 | 实时点云支撑 + 净空 |
| 落点 xy、端点位姿、关节解 | 搜索 + 控制器 IK |
| 轨迹 | MoveIt RRTConnect，**第 1 次尝试就过** |
| 下降 | 四步，单步 ≤12 mm、累计 ≤120 mm |

**示教只当先验，不回放任何路点**：候选排序中心、IK 种子、腕部姿态基准、一个担保候选，
外加接触实测支撑高度做交叉校验。见 `free_table_place.DemonstratedPlace`。

交叉验证（示教没参与高度计算）：

| | 算出来 | 示教实测 | 差 |
|---|---|---|---|
| 放置 TCP 高度 | −0.2018 | −0.2036 | **1.8 mm** |
| 落点桌面高度 | −0.3121 | −0.3143 | **2.2 mm** |

**人喂的那一个：`--bottle-bottom-below-tcp-m 0.12`**（见 §4）。

---

## 3. 修掉的十一处，全是同一个根因

这条管线是按**「货架 + TCP 就在瓶轴上」**写的。放置是**「桌面 + 水平抓，TCP 偏瓶轴 63 mm」**。

| # | 问题 | 症状 |
|---|---|---|
| 1 | `scene_image_bottom_crop=405` 当自体过滤器 | 近半张桌子（唯一够得着的那半）看不见 |
| 2 | 采样步长 6 | 拟合门槛是点数，被饿死 |
| 3 | 单一 z + 12 mm 内点带 | 桌面实际有 22 mm 斜度，只能咬住一段 |
| 4 | 候选按「离 ROI 中心多远」排序 | 完全不看可达性，前 20 名全超臂展 |
| 5 | IK 从收拢位起解 | 控制器 IK 认种子，整片候选 rc=1 |
| 6 | 碰撞场景含机器人自己 | `r_hand` 撞自己的点云 |
| 7 | **持瓶包络高 40 mm** | **29 次规划全败；修完第 1 次就过** |
| 8 | `placement_still_valid` 比 TCP | 恒报 63 mm 漂移 |
| 9 | 测量圆柱以 TCP 开（半径 49 < 偏移 63） | 整个错过瓶子，量到桌沿 |
| 10 | 释放确认比 TCP | 47.2 mm，把成功判成失败 |
| 11 | 中止路径硬写 `bottle_released: false` | 已松爪却记成没放（第 8 条的老坑） |

**第 7 条是钥匙**：`side_table_template` 里抄了一份通用瓶高 `held_bottle_height_m=0.255`，
而外星人是 **0.215**。包络底因此陷进桌面碰撞盒 4 mm，起始状态非法，
`planning_time=0.166s`（预算 6 s）—— 规划器根本没搜。现在改成读商品表。

> 教训就是第 7 条本身：**同一个量不要算两遍。** 商品几何只有一个家，是 `utils/items_info.py`。

---

## 4. 那个 120 mm：正解是量，不是推

`--bottle-bottom-below-tcp-m` 是**这次抓取的属性**，不是商品的属性 ——
抓取点是按检测框高度比例 `grasp_height_fraction=0.69` 选的，
所以商品表的 `height/2 = 108 mm` 只是「夹在瓶身中部」的假设。

**下一步应该把它记进抓取流程。但不要从场景几何推：**

```
source_support_surface_id = fence_shelf_bottom  → 顶面 -0.2325
抓取 TCP z                                      = -0.1546
推出来                                          =  78 mm     ← 错 42 mm
```

真值 ≈120 mm。反证：若真是 78 mm，最后 TCP 停在 −0.185 时瓶底在 −0.263，
离桌面还有 48 mm，瓶子会从 5 cm 摔下去；实际它稳稳立着（3/3 共识、散布 1.0 mm）。

推错的原因有两层，叠加成 42 mm：

- `bottle.pose.xyz` 存的是**抓取点高度**（`localization_to_mtc_scenario.py`：
  `bottle_center = surface_moveit + approach*radius`，只补了水平方向），不是瓶心
- 支撑面盒顶不等于瓶子真正站立的那个面

**该怎么做**：抓取抬起之后、瓶子悬在自由空间那一刻，用头部相机量一次瓶底，
写进 `pick_record.json` 的 `completion`。那时下方没有桌面漏进测量圆柱 ——
这正是桌面上做不到的条件（见 §5）。要改 `mtc_execution.execute_pick`
（它目前拿不到相机），**并且必须靠一次真机抓取验证**。

---

## 5. 桌面上为什么伺服不了

`_measure_held_bottle_bottom_z` 在桌面场景下不可信，实测证据：

```
第 1 次 相机读 88.9 mm
第 2 次 相机读 78.7 mm    ← 中间只下降了约 10.7 mm
```

偏移随下降**等量减小**，说明量到的是空间里不动的东西 —— 桌面本身。
`floor = 桌面 + 8 mm`，而这张桌子自己就斜 22 mm，局部误差轻易超过 8 mm。

所以给了 `--bottle-bottom-below-tcp-m` 时，观测**照做、只记录、不否决**，
下降走开环，但单步/累计上限、围栏、全链碰撞复核**一步不减**。
证据文件里 `offset_was_observed` 和 `operator_supplied_offset_m` 会写明。

---

## 6. 还没证明的（别当成已完成）

- **只成功过一次。** 决策链是确定性的、每步有证据，但「可复制」要第二次跑来挣
- **底盘和升降是人摆的**，脚本只校验不建立
- **抓取证据时效被关成 `inf`**（操作者授权，写进 `placement_only_execution.json`）
- **第 10、11 两处修复没在真机上验证过** —— 瓶子已放下，验它得再抓一次
- `bottle_bottom_below_tcp_m = 0.040` 仍留在 profile 里，加载器要求它存在但放置已不读
- 机器人上的 `place_held_bottle_only_20260813.py` 已被新签名弄坏，**不要再用**，
  用仓库里的 `scripts/place_held_bottle_only.py`

---

## 7. 只读探针

`scripts/probe_output_table_visibility.py` —— 不连机械臂、不动升降底盘，
一次采集给出：裁剪前后的 ROI 点数、平面拟合、候选、桌面 z 分带、
按格计数图和**按格中位高度图**（判斜面/碗面用的就是它）。

```bash
python3 scripts/probe_output_table_visibility.py \
  --preferred-xy 0.103 0.551 --min-patch-points 30 --stride 3
```
