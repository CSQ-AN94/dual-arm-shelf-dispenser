import xml.etree.ElementTree as ET

from grabber_robot_state_bridge.robot_description import (
    install_ctag2f90d_geometry,
)


def test_installed_ctag2f90d_replaces_both_fake_hands_with_sized_boxes():
    source = """<robot name="test">
      <link name="r_hand">
        <inertial/>
        <visual><geometry><mesh filename="old/r_hand_base_link.STL"/></geometry></visual>
        <collision><geometry><mesh filename="old/r_hand_base_link.STL"/></geometry></collision>
      </link>
      <link name="l_hand_link">
        <inertial/>
        <visual><geometry><mesh filename="old/l_hand_base_link.STL"/></geometry></visual>
        <collision><geometry><mesh filename="old/l_hand_base_link.STL"/></geometry></collision>
      </link>
      <link name="r_link7"/>
    </robot>"""

    root = ET.fromstring(install_ctag2f90d_geometry(source))
    for link_name in ("r_hand", "l_hand_link"):
        hand = root.find(f"./link[@name='{link_name}']")
        assert hand is not None
        assert hand.findall("./collision/geometry/mesh") == []
        assert [
            box.get("size") for box in hand.findall("./collision/geometry/box")
        ] == ["0.075 0.0508 0.131", "0.151 0.023 0.0915"]
        assert [
            item.get("xyz") for item in hand.findall("./collision/origin")
        ] == ["0 0 0.03820", "0 0 0.14945"]
        assert len(hand.findall("visual")) == 3
    assert root.find("./link[@name='r_link7']") is not None
