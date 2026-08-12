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


def test_a_stray_detection_frame_means_not_this_layer_not_a_fault():
    """One false-positive frame must not abort the whole layer search.

    2026-08-12: P04 was on the lower layer.  The detector produced a single
    spurious frame on the upper one, the 1/7 result was classified as a fault,
    and the search stopped without ever looking below.  The classifier keyed on
    "were there zero detector hits", which any non-zero false-positive rate
    defeats.  The floor that matters is the consensus test's own: below it no
    outcome could have been a confident detection.
    """
    import shelf_dispenser.orchestrator as orch

    assert orch.MINIMUM_CONSENSUS_FRAMES == 3
    source = Path(orch.__file__).read_text(encoding="utf-8")
    branch = source[source.index("检测到目标但稳定帧不足以确认")
                    - 1200: source.index("检测/深度稳定帧不足")]
    # Below the floor: BottleDetectionLost, which the layer runner catches and
    # turns into "try the next height".
    assert "if 0 < len(camera_points) < MINIMUM_CONSENSUS_FRAMES:" in branch
    # Zero usable points with a detector hit stays a fault: the detector saw
    # the product and depth produced nothing, which is the depth path
    # failing rather than an empty layer.
    assert "BottleDetectionLost" in branch
    # At or above it the fault classification stays, so a layer that really
    # holds the target is never silently skipped.
    assert "raise SafetyAbort" in source[source.index("检测/深度稳定帧不足") - 200:
                                         source.index("检测/深度稳定帧不足") + 200]
