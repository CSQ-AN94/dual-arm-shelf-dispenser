"""Console presentation: progress, phase timing, and evidence preservation.

The point of this module is that the operator's view and the forensic log
diverge.  These tests pin the parts where getting it wrong would either
hide a running motion (looks like a hang) or lose evidence.
"""

import io
import logging
import time

from bottle_grasp.console import (
    STAGE_FLAG,
    ConsoleFormatter,
    ProgressReporter,
    RunTimeline,
)


def _record(message: str, *, level=logging.INFO, stage=False):
    record = logging.LogRecord(
        "bottle_demo", level, __file__, 1, message, None, None
    )
    if stage:
        setattr(record, STAGE_FLAG, True)
    return record


def test_stage_and_detail_lines_are_visually_distinguishable():
    formatter = ConsoleFormatter(colour=False)
    stage = formatter.format(_record("移动到观察位", stage=True))
    detail = formatter.format(_record("轨迹 86 点"))
    assert stage.startswith("\n▸")
    assert "移动到观察位" in stage
    assert not detail.startswith("\n")
    assert "▸" not in detail


def test_stage_line_reports_elapsed_so_slowness_is_attributable():
    formatter = ConsoleFormatter(colour=False)
    formatter.format(_record("第一步", stage=True))
    time.sleep(0.05)
    second = formatter.format(_record("第二步", stage=True))
    assert "上一步" in second and "累计" in second


def test_warning_and_error_are_marked():
    formatter = ConsoleFormatter(colour=False)
    warning = formatter.format(_record("注意", level=logging.WARNING))
    error = formatter.format(_record("挂了", level=logging.ERROR))
    assert "⚠" in warning
    assert "✖" in error


def test_progress_without_a_tty_still_emits_records():
    """A piped/redirected run must not go silent for the whole motion."""
    logger = logging.getLogger("test.progress.pipe")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    seen = []
    logger.addHandler(
        type(
            "Capture",
            (logging.Handler,),
            {"emit": lambda _self, record: seen.append(record.getMessage())},
        )()
    )
    reporter = ProgressReporter(
        "轨迹执行",
        100,
        logger=logger,
        stream=io.StringIO(),  # not a tty
        min_interval_s=0.0,
    )
    reporter.update(25)
    reporter.close()
    assert any("25/100" in message for message in seen)
    assert any("完成" in message for message in seen)


def test_progress_reports_percentage_and_eta():
    logger = logging.getLogger("test.progress.eta")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    seen = []
    logger.addHandler(
        type(
            "Capture",
            (logging.Handler,),
            {"emit": lambda _self, record: seen.append(record.getMessage())},
        )()
    )
    reporter = ProgressReporter(
        "轨迹执行", 200, logger=logger, stream=io.StringIO(),
        min_interval_s=0.0,
    )
    time.sleep(0.02)
    reporter.update(50)
    assert any("25%" in message for message in seen)
    assert any("预计还需" in message for message in seen)


def test_progress_is_throttled_so_a_fast_loop_cannot_flood():
    logger = logging.getLogger("test.progress.throttle")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    seen = []
    logger.addHandler(
        type(
            "Capture",
            (logging.Handler,),
            {"emit": lambda _self, record: seen.append(record.getMessage())},
        )()
    )
    reporter = ProgressReporter(
        "轨迹执行", 1000, logger=logger, stream=io.StringIO(),
        min_interval_s=60.0,
    )
    for index in range(500):
        reporter.update(index)
    assert len(seen) <= 1


def test_timeline_ranks_the_slowest_phases():
    timeline = RunTimeline()
    timeline.mark("快的一步")
    time.sleep(0.01)
    timeline.mark("慢的一步")
    time.sleep(0.08)
    timeline.finish()
    rendered = timeline.render()
    assert "耗时分布" in rendered
    # The slow phase must be listed above the fast one.
    assert rendered.index("慢的一步") < rendered.index("快的一步")


def test_empty_timeline_renders_nothing():
    assert RunTimeline().render() == ""
