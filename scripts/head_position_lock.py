#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
头部舵机位置锁定/校验工具（人工诊断用命令行封装）。

协议与基准值现在统一定义在 `shelf_dispenser/head_lock.py`（2026-07-17起，
`shelf_dispenser/demo.py` 每次运行都会在开始前自动调用同一套逻辑强制回到基准
角度，不再需要人工在标定/demo 之间手动跑这个脚本）。这个命令行工具保留用于
现场人工诊断："机械臂做了什么之后头部感觉不对，查一下有没有漂移"。

用法:
  python3 scripts/head_position_lock.py check     # 只读当前角度，对比基准值
  python3 scripts/head_position_lock.py restore   # 如果偏了，自动微调回基准值
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser import head_lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "restore"])
    args = parser.parse_args()

    if args.action == "check":
        current = head_lock.read_current_angle()
        if not current:
            print("[FATAL] 没收到角度广播，head_servo_ctrl.py 是否在运行？")
            return 1
        ok = head_lock.is_at_reference(current)
        d1 = current["angle1"] - head_lock.HEAD_REFERENCE["angle1"]
        d2 = current["angle2"] - head_lock.HEAD_REFERENCE["angle2"]
        print(
            f"当前: {current}  基准: {head_lock.HEAD_REFERENCE}  "
            f"偏差: angle1={d1:+d} angle2={d2:+d}"
        )
        print("状态: 未漂移" if ok else "状态: 已漂移！先 restore")
        return 0 if ok else 1

    result = head_lock.restore_reference()
    if result["ok"]:
        print(f"恢复完成 @ step {result['steps']}: {result['angle']}")
        return 0
    print(f"[WARN] 恢复失败: {result.get('reason')}，当前: {result.get('angle')}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
