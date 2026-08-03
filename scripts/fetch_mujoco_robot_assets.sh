#!/usr/bin/env bash
set -euo pipefail

# Read-only asset fetch.  This never imports the robot SDK or starts ROS.
robot_host="${ROBOT_HOST:-rm@192.168.3.68}"
remote_description="${ROBOT_DESCRIPTION_DIR:-/home/rm/ros2_ws/install/dual_rm_75b_description/share/dual_rm_75b_description}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${repo_root}/.mujoco_assets/dual_rm_75b_description"

mkdir -p "${destination}"
rsync -az "${robot_host}:${remote_description}/meshes/" "${destination}/meshes/"
rsync -az "${robot_host}:${remote_description}/urdf/" "${destination}/urdf/"
ssh "${robot_host}" \
  "bash -lc 'source /opt/ros/humble/setup.bash; source /home/rm/ros2_ws/install/setup.bash; xacro /home/rm/ros2_ws/install/dual_rm_75b_moveit_config/share/dual_rm_75b_moveit_config/config/dual_rm_75b_description.urdf.xacro'" \
  > "${destination}/urdf/dual_rm_75b_moveit_expanded.urdf"

test -s "${destination}/urdf/dual_rm_75b_moveit_expanded.urdf"
test -f "${destination}/meshes/r_link7.STL"
test -f "${destination}/meshes/rmg24_finger1_link.STL"
echo "MuJoCo robot assets ready: ${destination}"
