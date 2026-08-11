#!/usr/bin/env python3
"""侧桌投放 profile 还缺什么 —— 纯离线，不碰机器人。

`load_safety_profile` 在 `enabled: false` 时第一步就 abort，所以它看不到里面还
缺哪些字段；`_validated_shelf_ready` 一次只报一个缺失项，在现场意味着填一项重
跑一次。这个脚本直接读 json，一次把全部缺口列出来，给现场标定当填表清单用。

    python3 scripts/side_table_profile_status.py

缺口全部补齐时退出码 0，还有 null / false 时退出码 1。每项该怎么量、量出来的
数该往哪填，见 docs/side_table_delivery_reopening.md。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILES = ROOT / "shelf_dispenser" / "safety_profiles.json"
PROFILE_NAME = "side_table_template"


def walk(node, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten the profile subtree into (dotted path, leaf value) pairs.

    Lists are descended into, not treated as leaves: `allowed_tcp_zones` and
    `keepout_boxes` are lists of boxes whose min/max are the actual nulls, and
    a non-empty list of empty boxes would otherwise read as "filled".  A
    numeric triple like `table_roi.min` is a leaf, though -- descending into
    it would report three nulls for one measurement.
    """
    if isinstance(node, dict):
        return [
            pair
            for key, value in node.items()
            for pair in walk(value, f"{prefix}.{key}" if prefix else key)
        ]
    if isinstance(node, list) and any(isinstance(item, (dict, list)) for item in node):
        return [
            pair
            for index, item in enumerate(node)
            for pair in walk(item, f"{prefix}[{index}]")
        ]
    return [(prefix, node)]


def main() -> int:
    raw = json.loads(PROFILES.read_text(encoding="utf-8"))
    profile = raw["profiles"][PROFILE_NAME]
    delivery = profile.get("side_table_delivery")
    if delivery is None:
        print(f"{PROFILE_NAME} 没有 side_table_delivery 段", file=sys.stderr)
        return 2

    # The two profile-level switches load_safety_profile checks before it ever
    # reaches these fields.  They are the last thing to flip, not the first.
    switches = [
        ("enabled", profile.get("enabled")),
        ("verified_for_execution", profile.get("verified_for_execution")),
    ]
    # The arm cannot plan against the post-turn table at all until the
    # profile-level envelope is measured too, and those live outside
    # side_table_delivery.  Leaving them out here would just move the
    # surprise to the next on-site run.
    leaves = walk(delivery)
    for key in ("tcp_workspace", "allowed_tcp_zones", "keepout_boxes"):
        leaves += walk(profile.get(key), key)
    # A confirmation flag left false is a gap even though it is not null.
    missing = [
        (path, value)
        for path, value in leaves
        if value is None or (path.endswith(("_verified", ".verified")) and value is not True)
    ]
    filled = [pair for pair in leaves if pair not in missing]

    print(f"profile: {PROFILE_NAME}  ({PROFILES.relative_to(ROOT)})")
    for name, value in switches:
        print(f"  {name}: {value}")
    print(f"\n已填 {len(filled)} 项：")
    for path, value in filled:
        print(f"  {path} = {value!r}")
    print(f"\n还缺 {len(missing)} 项：")
    for path, value in missing:
        note = "确认位仍为 false" if value is not None else "null"
        print(f"  {path}  ({note})")

    if missing:
        print(
            "\n还有缺口，入口保持关闭。量测方法见 "
            "docs/side_table_delivery_reopening.md",
            file=sys.stderr,
        )
        return 1
    print("\n全部字段已填且确认位为 true。可以考虑翻 enabled/verified_for_execution。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
