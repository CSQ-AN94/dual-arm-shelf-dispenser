"""Replace the vendor's fake five-finger hands with the installed grippers."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET


MOUNT_YAW_RAD = math.pi / 2.0
_MOUNT_RPY = f"0 0 {MOUNT_YAW_RAD}"
# r_hand_joint / l_hand_link_joint in dual_rm_75b_description/urdf/joint.urdf.xacro:
# a fixed 44.5 mm along +Z of link7, no rotation.  Named here because the
# geometry below is expressed relative to it and offline tools need the same
# number to place these boxes from a TCP pose.
HAND_LINK_Z_FROM_IK_LINK_M = 0.0445
_HAND_LINKS = {
    "r_hand": "r_hand_base_link.STL",
    "l_hand_link": "l_hand_base_link.STL",
}

# CTAG2F90D manufacturer drawing (2025-08-18): 222.5 mm overall length,
# 151 mm maximum outer width, 90 mm clear opening, 75 mm palm width,
# 45 mm palm depth and 23 mm finger depth. The palm-to-knuckle transition is
# not dimensioned; 131 mm is read conservatively from the scaled front view.
# Keep it as a calibration knob until the official CAD is admitted here.
# https://imgs-data-brwq.bcdn8.com/zhixing1116/uploads/20260108/6da9ea4446e76f98f4205b954eb47d0a.pdf
_PALM_LENGTH_M = 0.131
_FINGER_LENGTH_M = 0.2225 - _PALM_LENGTH_M

# The installed hand-link frame is 44.5 mm beyond link7, while the measured
# controller flange is 17.2 mm beyond link7. Preserve the existing link frame
# and shift only the geometry back by their 27.3 mm difference.
_BASE_Z_M = 0.0172 - HAND_LINK_Z_FROM_IK_LINK_M
_PALM_CENTER_Z_M = _BASE_Z_M + _PALM_LENGTH_M / 2.0
_FINGER_CENTER_Z_M = _BASE_Z_M + _PALM_LENGTH_M + _FINGER_LENGTH_M / 2.0

# Public so offline collision diagnostics read the same boxes MoveIt is given
# rather than a second copy that can drift from it.
COLLISION_BOXES = (
    ("0.075 0.0508 0.131", f"0 0 {_PALM_CENTER_Z_M:.5f}"),
    # MoveIt cannot observe finger position, so avoidance covers the entire
    # maximum-open swept volume. This is deliberately solid across the gap.
    ("0.151 0.023 0.0915", f"0 0 {_FINGER_CENTER_Z_M:.5f}"),
)

_FINGER_WIDTH_M = (0.151 - 0.090) / 2.0
_FINGER_OFFSET_M = 0.090 / 2.0 + _FINGER_WIDTH_M / 2.0
_VISUAL_BOXES = (
    COLLISION_BOXES[0],
    (
        f"{_FINGER_WIDTH_M:.4f} 0.023 0.0915",
        f"0 {-_FINGER_OFFSET_M:.5f} {_FINGER_CENTER_Z_M:.5f}",
    ),
    (
        f"{_FINGER_WIDTH_M:.4f} 0.023 0.0915",
        f"0 {_FINGER_OFFSET_M:.5f} {_FINGER_CENTER_Z_M:.5f}",
    ),
)


def _add_boxes(link, tag, boxes):
    for size, xyz in boxes:
        element = ET.SubElement(link, tag)
        ET.SubElement(element, "origin", {"xyz": xyz, "rpy": _MOUNT_RPY})
        geometry = ET.SubElement(element, "geometry")
        ET.SubElement(geometry, "box", {"size": size})


def install_ctag2f90d_geometry(robot_xml: str) -> str:
    """Install conservative CTAG2F90D geometry without changing kinematics.

    Existing hand link names and joints stay untouched so the SRDF, touch
    links and audited TCP transform remain valid. Collision geometry uses the
    fully-open swept envelope; split visuals only make RViz recognizable as a
    two-finger gripper.
    """

    root = ET.fromstring(robot_xml)
    for link_name, obsolete_mesh in _HAND_LINKS.items():
        hands = [
            link for link in root.findall("link") if link.get("name") == link_name
        ]
        if len(hands) != 1:
            raise RuntimeError(
                f"expected exactly one {link_name} link, found {len(hands)}"
            )
        hand = hands[0]
        old_meshes = [
            mesh.get("filename", "")
            for collision in hand.findall("collision")
            for mesh in collision.findall("./geometry/mesh")
        ]
        if len(old_meshes) != 1 or not old_meshes[0].endswith(obsolete_mesh):
            raise RuntimeError(
                f"unexpected {link_name} collision geometry: {old_meshes}"
            )

        for tag in ("visual", "collision"):
            for element in hand.findall(tag):
                hand.remove(element)
        _add_boxes(hand, "visual", _VISUAL_BOXES)
        _add_boxes(hand, "collision", COLLISION_BOXES)
    return ET.tostring(root, encoding="unicode")
