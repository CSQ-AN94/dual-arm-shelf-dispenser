# Grabber Demo 与真机操作指南

本文档把当前仓库里的 demo、真机安全规则、RealMan 后台进程冲突、ROS/相机/夹爪参考命令整理到一起。README 只保留入口和概览；实际跑机器人前请先读这里。

## 1. 当前推荐路径

当前仓库有两条路径：

1. 应用主路径：`direct_grab.py`
   - 用 YOLO + RealSense 扫描商品。
   - 操作者输入商品名或编号。
   - 调用确定性抓取流程执行抓取。
   - 适合验证完整应用链路，但依赖 `config.yaml` 标定、相机、模型和机械臂配置。

2. 真机单步 demo：`grasp_demo.py` / `grasp_demo_left.py`
   - 不依赖 YOLO、RealSense、Gemini 或语音。
   - 只验证一组固定 HOME / 预抓取 / 抓取位姿。
   - 使用“停遥操 -> 纯 SDK 抓取 -> 官方 upstart 重启遥操”的安全架构。
   - 适合先确认机械臂、夹爪、遥操恢复这条最小链路。



## 2. 重要后台进程

真机上最容易出问题的是控制权冲突。机器人电脑后台常驻两个进程：

```text
atom            关节遥操控制器。它会约 100Hz 向机械臂持续发送关节保持/遥操指令。
zhixing_ctrl.py 夹爪遥操控制器。它会约 10Hz 向夹爪持续发送开合位置指令。
```

这两个进程的作用是让机器人在遥操模式下保持可控：

```text
atom            负责机械臂关节，不停告诉从臂当前应该保持/跟随到哪里。
zhixing_ctrl.py 负责夹爪，不停告诉夹爪当前应该开到什么位置。
```

但是跑 SDK、ROS 或 TCP demo 时，程序也会向机械臂和夹爪发命令。如果后台遥操进程还在运行，就会出现两个控制源同时发命令：

```text
程序发 movej / movel          atom 又立刻发关节保持命令覆盖它。
程序发 gripper open / close   zhixing_ctrl.py 又立刻发夹爪遥操命令覆盖它。
```

所以跑抓取 demo 前，通常要先停止它们，让 SDK 独占控制权：

```bash
pkill -x atom
pkill -f zhixing_ctrl.py
```

`grasp_demo.py` 和 `grasp_demo_left.py` 会自动做这件事，并在结束时调用官方 `upstart_all.sh` 恢复遥操。不要随便用 `SIGSTOP` 冻住 `atom` 后直接恢复遥操，因为主从位置可能错位，恢复瞬间可能产生危险动作。

检查它们是否在运行：

```bash
ps aux | grep -E 'atom|zhixing' | grep -v grep
```

## 3. 机器人与环境信息

远程机器人电脑：

```bash
ssh rm@192.168.3.68
password: rm
```

机械臂 TCP：

```text
左臂: 169.254.128.18:8080
右臂: 169.254.128.19:8080
```

ROS2：

```text
ROS_DISTRO=humble
工作空间: ~/ros2_ws
右臂 driver 配置:
/home/rm/ros2_ws/install/dual_rm_driver/share/dual_rm_driver/config/dual_75_right_config.yaml
```

相机/LeRobot 环境：

```bash
cd ~/Dev/bi_realman_ws/third_party/lerobot_robot_bi_realman
conda activate lerobot
```

Grabber 仓库运行环境优先使用机器人主机 Python 环境。Docker 目前只是开发容器草案，还没有完整验证 RealSense、Realman SDK、GPU、USB/CAN 权限组合。

## 4. 安全规则

开始前确认：

```text
1. 机械臂周围没有人手、线缆、工具、杯子等障碍物。
2. 急停按钮在手边。
3. 遥操、示教器、网页控制、手柄控制处于空闲或已停止。
4. 第一次验证时只调一侧手臂，不要同时跑左右臂抓取。
5. 夹爪动作不要夹到手、线缆或硬物。
6. 跑升降和头部舵机测试前，确认相机线、头部结构、升降范围内没有遮挡。
```

机器人电脑上有两个关键后台进程：

```text
atom            关节遥操控制器，约 100Hz 向从臂发 CANFD 位置保持指令。
zhixing_ctrl.py 夹爪遥操控制器，约 10Hz 向夹爪发位置命令。
```

如果它们不让出控制权，SDK、ROS 或 TCP 发出的运动/夹爪命令可能被覆盖。最常见表现是：命令返回但手臂不动、夹爪只动一点、控制器 busy、夹爪阻塞命令超时，严重时需要断电重启恢复夹爪控制器。

查看进程：

```bash
ps aux | grep -E 'atom|zhixing' | grep -v grep
```

## 5. Demo 速查

### `pose_reader.py`

用途：遥操机械臂时实时读取关节角和末端位姿，不停止 `atom`，用于记录 HOME 和抓取点。

右臂：

```bash
python3 pose_reader.py 169.254.128.19
```

左臂：

```bash
python3 pose_reader.py 169.254.128.18
```

记录两类值：

```text
HOME_JOINTS: 7 个关节角，单位 deg。
GRASP_POSE : [x, y, z, rx, ry, rz]，单位 m/rad。
```

### `grasp_demo.py`

用途：右臂单次固定点抓取。

```bash
python3 grasp_demo.py
```

可手动指定 IP/端口：

```bash
python3 grasp_demo.py 169.254.128.19 8080
```

流程：

```text
1. pkill atom 和 zhixing_ctrl.py，让出控制权。
2. 连接 Realman SDK。
3. 初始化 Realman Plus 夹爪通信。
4. 夹爪闭合到默认状态。
5. 回 HOME。
6. 到预抓取位姿。
7. 抓取前打开夹爪。
8. 直线进入抓取位姿。
9. 闭合夹爪。
10. 抬起并回 HOME。
11. 松开物体。
12. 夹爪恢复默认闭合。
13. SDK 断开，用官方 upstart 重启遥操并验证双臂错误码。
14. 需要同时按遥操上的两个按钮来恢复遥操
```



### `grasp_demo_left.py` (目前左臂的两个位姿没有填入)

用途：左臂单次固定点抓取模板。

第一次使用前先读左臂位姿：

```bash
python3 pose_reader.py 169.254.128.18
```

然后填入：

```python
LEFT_POSES_VERIFIED = True
HOME_JOINTS = [...]
GRASP_POSE = [...]
```

再运行：

```bash
python3 grasp_demo_left.py
```

左臂和右臂的 HOME / 抓取位姿不能混用。

### `direct_grab.py`

用途：当前应用主入口。

交互模式：

```bash
python3 direct_grab.py
```

一次性扫描：

```bash
python3 direct_grab.py scan
```

一次性抓取：

```bash
python3 direct_grab.py grab 红牛
```

交互命令：

```text
scan        重新扫描
红牛         抓取识别列表中的“红牛”
1           抓取当前识别列表第 1 个商品
抓 红牛      同上
q           退出
```

注意：`direct_grab.py` 依赖 `config.yaml` 中的双臂 IP、`active_arm`、RealSense 序列号、YOLO 模型路径、手眼矩阵和抓取/放置位姿。当前这些配置仍有占位值，上真实货架前必须重新标定。

### `sdk_demo.py`

用途：低层 SDK 连通测试，可测左右臂关节和夹爪。

```bash
python3 sdk_demo.py
python3 sdk_demo.py left
python3 sdk_demo.py right
```

这不是推荐的完整抓取流程。它会对 `atom` / `zhixing_ctrl.py` 做 SIGSTOP/SIGCONT，用于硬件连通和关节方向确认。做真实抓取时优先参考 `grasp_demo.py` 的停遥操和官方恢复方式。

### `test/` 硬件测试脚本

`test/` 目录里是低层硬件验证脚本，不属于 `direct_grab.py` 主抓取流程。它们适合在正式抓取前单独确认升降机构和头部舵机是否可控。

当前文件：

```text
test/test_lift.py         升降机构 TCP 测试
test/test_head_servo.py   头部舵机串口测试
test/run_head_servo.sh    用机器人本机 Python 启动头部舵机测试
```

#### `test/test_lift.py`

用途：测试升降机构。脚本连接 `169.254.128.18:8080`，先读取当前高度，然后交互确认后依次执行：

```text
1. 下降 100 mm。
2. 上升 50 mm。
3. 恢复到测试前的原始高度。
4. 再次读取高度确认。
```

运行：

```bash
python3 test/test_lift.py
```

注意：

```text
1. 这个脚本会真实移动升降机构。
2. 每一步动作前都会要求按 Enter 确认，可以用 Ctrl+C 取消。
3. 下降目标不会低于 0 mm，上升目标不会超过 2600 mm。
4. 开始前确认升降方向没有障碍物，线缆不会被拉扯。
```

#### `test/test_head_servo.py`

用途：测试头部两个舵机 ID 和方向。脚本直接操作舵机串口，默认端口是：

```text
/dev/rmUSB3
```

也可以在脚本里改成 `/dev/ttyUSB0`。当前假设：

```text
ID=1  Pitch，抬头/低头，范围 400~600，中位 500。
ID=2  Yaw，左转/右转，范围 200~800，中位 500。
```

推荐在机器人本机运行：

```bash
sh test/run_head_servo.sh
```

如果当前 Python 环境已经有 `serial` / `pyserial`，也可以直接运行：

```bash
python3 test/test_head_servo.py
```

交互命令：

```text
c       回中到 500, 500
u / d   按工程假设测试抬头 / 低头
l / r   按工程假设测试左转 / 右转
1+ 1-   直接测试 ID=1 正/负方向
2+ 2-   直接测试 ID=2 正/负方向
read    只读当前角度
scan    只读扫描 ID 0~15
q       退出
```

注意：

```text
1. 脚本启动时只读取当前角度，不会自动回中。
2. 退出时也不会自动回中；需要回中就先输入 c。
3. 如果读数变化但肉眼不明显，可能是幅度小、方向假设不对，或相机不在对应轴上。
4. 如果目标变了但读数几乎不变，可能是串口被占用、ID 不对、舵机卡住或命令没生效。
```

这个 ID=1 的舵机应该存在问题, 摄像头不可以上下移动, 只可以左右移动.

## 6. ROS 右臂 driver 与夹爪参考

ROS 控制不属于当前 Grabber 主路径，但对排查 driver、topic 和夹爪链路很有用。

每个远程终端先执行：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

启动右臂 driver：

```bash
/home/rm/ros2_ws/install/dual_rm_driver/lib/dual_rm_driver/dual_rm_driver \
  --ros-args \
  -r __ns:=/right_arm_controller \
  --params-file /home/rm/ros2_ws/install/dual_rm_driver/share/dual_rm_driver/config/dual_75_right_config.yaml
```

确认节点：

```bash
ros2 node list
```

期望看到：

```text
/right_arm_controller/rm_driver
/right_arm_controller/udp_publish_node
```

夹爪 topic：

```text
命令: /right_arm_controller/rm_driver/set_gripper_position_cmd
结果: /right_arm_controller/rm_driver/set_gripper_position_result
类型: rm_ros_interfaces/msg/Gripperset
```

监听结果：

```bash
ros2 topic echo /right_arm_controller/rm_driver/set_gripper_position_result
```

发送打开/闭合/半开：

```bash
ros2 topic pub --once /right_arm_controller/rm_driver/set_gripper_position_cmd rm_ros_interfaces/msg/Gripperset \
"{position: 1000, block: false, timeout: 5}"

ros2 topic pub --once /right_arm_controller/rm_driver/set_gripper_position_cmd rm_ros_interfaces/msg/Gripperset \
"{position: 1, block: false, timeout: 5}"

ros2 topic pub --once /right_arm_controller/rm_driver/set_gripper_position_cmd rm_ros_interfaces/msg/Gripperset \
"{position: 500, block: false, timeout: 5}"
```

重要：发 ROS 夹爪命令前必须停止 `zhixing_ctrl.py`。如果还要控制关节，也要停止 `atom`。

```bash
pkill -f zhixing_ctrl.py
pkill -x atom
```

优先使用 `block: false`。如果 `block: true` 时后台遥操仍在发夹爪命令，超时后可能导致夹爪控制器锁死。

## 7. 读取右臂状态

状态查询命令 topic：

```text
/right_arm_controller/rm_driver/get_current_arm_state_cmd
```

结果 topic：

```text
/right_arm_controller/rm_driver/get_current_arm_original_state_result
```

先监听结果：

```bash
ros2 topic echo /right_arm_controller/rm_driver/get_current_arm_original_state_result --once
```

再发查询：

```bash
ros2 topic pub --once /right_arm_controller/rm_driver/get_current_arm_state_cmd std_msgs/msg/Empty "{}"
```

当前不要用 `/joint_states` 判断真实关节角，因为它可能全是 0。以上 result topic 里的 `joint` 更可靠。

## 8. RealSense 相机快照

进入相机项目：

```bash
cd ~/Dev/bi_realman_ws/third_party/lerobot_robot_bi_realman
conda activate lerobot
```

确认环境：

```bash
which python3
python3 -m pip show lerobot
python3 -m pip show pyrealsense2
```

检查设备：

```bash
python3 - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.query_devices()
print("RealSense device count:", len(devices))
for i, dev in enumerate(devices):
    print(i, dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))
PY
```

拍快照：

```bash
python3 scripts/capture_realsense_snapshot.py \
  --output_dir=outputs/captured_images \
  --warmup_frames=5
```

从 Mac 拉取图片( )：

```bash
mkdir -p ~/Downloads/realman_captured_images

scp -r rm@192.168.3.68:/home/rm/Dev/bi_realman_ws/third_party/lerobot_robot_bi_realman/outputs/captured_images/* \
  ~/Downloads/realman_captured_images/

open ~/Downloads/realman_captured_images
```

## 9. 常见问题

### 命令发出但手臂不动

通常是 `atom` 仍在以高频位置保持覆盖外部命令。先确认当前应该是遥操模式还是程序控制模式，不要两边同时控制。

### 夹爪只动一点或没有结果

先确认 `zhixing_ctrl.py` 已停止，并监听 result topic 或 SDK 返回值。夹爪位置可以用 1 和 1000 做最大幅度测试。

### `ros2 topic pub` 只显示 publishing

这通常不是报错，只表示命令发出去了。需要另一个终端监听对应 result topic。

### 相机脚本找不到 `lerobot`

多数情况是没有进入 conda 环境：

```bash
conda activate lerobot
which python3
```

期望 Python 路径来自：

```text
/home/rm/miniconda3/envs/lerobot/bin/python3
```

### 头部舵机串口打不开

先确认是在机器人本机运行，并检查 `/dev/rmUSB3` 是否存在。如果不存在，可以看是否有 `/dev/ttyUSB0`，再把 `test/test_head_servo.py` 里的 `PORT` 改成对应设备。

### 升降机构没有高度返回

确认机器人网络能连到 `169.254.128.18:8080`，并且升降控制器没有被其他程序占用。脚本没有读到 `height` 会直接退出，不会继续发运动命令。
