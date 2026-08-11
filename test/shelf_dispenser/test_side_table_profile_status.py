"""侧桌 profile 填表清单的自检。

这个脚本只有两处真逻辑：把嵌套 profile 摊平成叶子，以及判定什么算「缺」。
两处都错得很安静 —— 摊平时把一列空盒子的 list 当成叶子，或者只看 null 不看
确认位，都会让清单显示「齐了」而现场照样 abort。所以各留一个断言。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "side_table_profile_status", ROOT / "scripts" / "side_table_profile_status.py"
)
status = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = status
_spec.loader.exec_module(status)


def test_walk_descends_into_lists_of_boxes_but_not_numeric_triples():
    leaves = dict(
        status.walk(
            {
                "keepout_boxes": [{"id": "a", "min": None, "max": [1.0, 2.0, 3.0]}],
                "table_roi": {"min": [0.1, 0.2, 0.3]},
            }
        )
    )
    # A list of boxes is structure: its nulls have to surface individually.
    assert leaves["keepout_boxes[0].min"] is None
    assert leaves["keepout_boxes[0].max"] == [1.0, 2.0, 3.0]
    # A numeric triple is one measurement, not three.
    assert leaves["table_roi.min"] == [0.1, 0.2, 0.3]


def test_the_shipped_template_is_reported_incomplete(capsys):
    """The tripwire: as long as the side table is closed, this must exit 1."""
    assert status.main() == 1
    out = capsys.readouterr().out
    assert "enabled: False" in out
    # A false confirmation flag is a gap even though it is not null.
    assert "rotation_sweep.positive.verified" in out
    assert "table_roi.min" in out


def test_measured_values_already_in_the_profile_read_as_filled(capsys):
    """+90 deg CCW was measured on the chassis; it is not a remaining gap."""
    status.main()
    out = capsys.readouterr().out
    filled, _, remaining = out.partition("还缺")
    assert "body_rotation_yaw_deg = 90.0" in filled
    assert "body_rotation_yaw_deg" not in remaining
