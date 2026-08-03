#!/usr/bin/env python3
"""Measure the package's shape so "is the architecture right" has an answer.

Four properties you can actually check, rather than opinions about layout:

  1. No import cycles.  A cycle means two modules cannot be understood, tested
     or replaced separately, and it never gets better on its own.
  2. A stable foundation.  Whatever most modules depend on must itself depend
     on little, or every change ripples.
  3. Dependencies point one way.  Fan-out belongs at the top (orchestrators),
     fan-in at the bottom (primitives).  A module with both is doing two jobs.
  4. No god module.  One file holding a large share of the code is where
     unrelated concerns get to touch each other -- which is exactly how the
     shelf pipeline ended up constrained by a table demo's home pose.

Run it after any structural change:  python scripts/architecture_report.py
Exit code is non-zero if a hard invariant (cycles) is broken.
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "shelf_dispenser"

# A single file over this share of the package is flagged.  Not a law -- a
# prompt to ask what else is living in there.
GOD_MODULE_SHARE = 0.15


def build_graph() -> tuple[list[str], dict[str, set[str]], dict[str, int]]:
    modules = sorted(p.stem for p in PACKAGE.glob("*.py") if p.stem != "__init__")
    known = set(modules)
    edges: dict[str, set[str]] = defaultdict(set)
    lines: dict[str, int] = {}
    for name in modules:
        source = (PACKAGE / f"{name}.py").read_text(encoding="utf-8")
        lines[name] = len(source.splitlines())
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            if node.module:
                target = node.module.split(".")[0]
                if target in known:
                    edges[name].add(target)
            else:
                edges[name].update(a.name for a in node.names if a.name in known)
    return modules, edges, lines


def find_cycles(modules: list[str], edges: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's strongly connected components; any component > 1 node is a cycle."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    found: list[list[str]] = []
    counter = 0

    def visit(node: str) -> None:
        nonlocal counter
        index[node] = low[node] = counter
        counter += 1
        stack.append(node)
        on_stack.add(node)
        for nxt in edges.get(node, ()):
            if nxt not in index:
                visit(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in on_stack:
                low[node] = min(low[node], index[nxt])
        if low[node] == index[node]:
            component = []
            while True:
                item = stack.pop()
                on_stack.discard(item)
                component.append(item)
                if item == node:
                    break
            if len(component) > 1:
                found.append(sorted(component))

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * len(modules) + 100))
    for name in modules:
        if name not in index:
            visit(name)
    return found


def main() -> int:
    modules, edges, lines = build_graph()
    total = sum(lines.values())
    fan_in: dict[str, int] = defaultdict(int)
    for source, targets in edges.items():
        for target in targets:
            fan_in[target] += 1

    print(f"shelf_dispenser/  {len(modules)} 个模块，{total} 行\n")

    cycles = find_cycles(modules, edges)
    print("① 循环依赖")
    if cycles:
        for cycle in cycles:
            print(f"   ✗ {' → '.join(cycle)} → {cycle[0]}")
    else:
        print("   ✓ 无")

    print("\n② 底座是否稳定（被依赖多、自己依赖少）")
    ranked = sorted(modules, key=lambda n: (-fan_in[n], lines[n]))
    for name in ranked[:5]:
        out = len(edges.get(name, ()))
        mark = "✓" if out <= 2 else "✗"
        print(
            f"   {mark} {name:24s} 被 {fan_in[name]:2d} 个依赖，"
            f"自己依赖 {out}，{lines[name]} 行"
        )

    print("\n③ 依赖是否单向（同时高扇入高扇出的模块在做两件事）")
    both = [
        n for n in modules if fan_in[n] >= 2 and len(edges.get(n, ())) >= 3
    ]
    if both:
        for name in both:
            print(
                f"   ✗ {name:24s} 被 {fan_in[name]} 个依赖，"
                f"又依赖 {len(edges[name])} 个"
            )
    else:
        print("   ✓ 没有")

    print("\n④ 有没有上帝模块")
    flagged = [n for n in modules if lines[n] > total * GOD_MODULE_SHARE]
    if flagged:
        for name in flagged:
            share = lines[name] / total
            print(
                f"   ✗ {name:24s} {lines[name]} 行，占全包 {share:.0%}，"
                f"被 {fan_in[name]} 个模块依赖"
            )
    else:
        print(f"   ✓ 没有单个模块超过全包 {GOD_MODULE_SHARE:.0%}")

    orphans = [n for n in modules if not edges.get(n) and not fan_in[n]]
    if orphans:
        print(
            "\n未通过 import 相连的模块（本项目里这些是 ROS 子进程入口，"
            "由路径调用，不是死代码）："
        )
        print("   " + ", ".join(orphans))

    return 1 if cycles else 0


if __name__ == "__main__":
    raise SystemExit(main())
