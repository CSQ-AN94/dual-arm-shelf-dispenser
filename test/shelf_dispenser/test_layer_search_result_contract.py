"""The layer search hands its answer back through a marked stdout line.

This broke on the first real run: the searcher printed a bare number, but
CameraThread writes its RealSense banner with a plain print(), so the shell
captured the banner concatenated with the height and refused to launch.  The
guard did its job -- a garbage lift height never reached the body -- but the
contract between the two sides was wrong, and only a live robot showed it.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEARCHER = ROOT / "scripts" / "find_product_shelf_layer.py"
RUN_TASK = ROOT / "scripts" / "run_task.sh"

# Verbatim from the 2026-08-12 run that failed.
POLLUTED_STDOUT = """\
[CameraThread] RealSense (153122071777) 初始化成功 848x480@30fps depth_scale=0.001000000m/unit
[CameraThread] 内参: fx=609.3 fy=609.7 cx=420.1 cy=247.6
[CameraThread] Pipeline stopped
SHELF_LAYER_LIFT_MM=250
[CameraThread] Pipeline stopped
"""

EXTRACT = (
    r"tr -d '\r' | sed -n 's/^SHELF_LAYER_LIFT_MM=\([0-9][0-9]*\)$/\1/p' | tail -1"
)


def _extract(text: str) -> str:
    return subprocess.run(
        ["bash", "-c", EXTRACT],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_the_height_survives_camera_banner_on_the_same_stream():
    assert _extract(POLLUTED_STDOUT) == "250"


def test_nothing_is_extracted_when_no_layer_was_selected():
    """An empty result must stay empty so the caller's own check can refuse."""
    assert _extract("[CameraThread] Pipeline stopped\n") == ""


def test_a_marker_with_junk_appended_is_not_accepted():
    """Anchored on both ends: a partially written line is not a height."""
    assert _extract("SHELF_LAYER_LIFT_MM=250abc\n") == ""
    assert _extract("prefixSHELF_LAYER_LIFT_MM=250\n") == ""


def test_both_sides_still_agree_on_the_marker():
    """Guards the split brain that caused the failure in the first place."""
    searcher = SEARCHER.read_text(encoding="utf-8")
    launcher = RUN_TASK.read_text(encoding="utf-8")
    assert 'RESULT_MARKER = "SHELF_LAYER_LIFT_MM="' in searcher
    assert 'print(f"{RESULT_MARKER}{height}", flush=True)' in searcher
    assert "SHELF_LAYER_LIFT_MM=" in launcher
