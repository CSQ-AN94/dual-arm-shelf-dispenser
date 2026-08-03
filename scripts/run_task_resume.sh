#!/usr/bin/env bash
set -euo pipefail

echo "此 launcher 已停用：新的观察位流程不会读取历史定位，也不是 resume。" >&2
echo "右臂已在观察位时请只运行：" >&2
echo "  scripts/run_task.sh from-observation" >&2
exit 2
