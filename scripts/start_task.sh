#!/usr/bin/env bash
set -euo pipefail

echo "此通用 launcher 已停用；请选择且只选择一个完整任务入口：" >&2
echo "  scripts/run_task.sh from-observation" >&2
echo "  scripts/run_task.sh from-start" >&2
exit 2
