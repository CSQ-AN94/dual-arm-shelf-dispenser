"""Terminal presentation for real-robot runs.

The run log and the terminal serve two different readers and must not be
formatted the same way:

- ``run.log`` / ``latest.log`` is forensic evidence.  It keeps every record,
  full timestamps and full detail, and is what gets replayed after a failure.
- The terminal is watched live by an operator with a hand on the e-stop.  It
  needs phase structure, visible progress during long motions, and enough
  timing to tell "still working" from "stuck".

2026-07-18 real run: after ``SDK 执行 MoveIt 轨迹: 146 个密集关节点`` the
terminal printed nothing for 90 seconds while the arm stepped through the
path.  Nothing was wrong, but from the outside it was indistinguishable from
a hang.  Long operations must report progress.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

# Marks a record that names a workflow phase rather than a detail line.
STAGE_FLAG = "bottle_stage"


def _supports_colour(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


class ConsoleFormatter(logging.Formatter):
    """Compact, scannable console lines; the file handler keeps the raw form.

    Phase lines are indented left and marked so the eye can find them in a
    scrolling terminal; detail lines sit indented under the phase they belong
    to.  Date and milliseconds are dropped here only — the file keeps them.
    """

    def __init__(self, *, colour: bool):
        super().__init__()
        self.colour = colour
        self._run_started = time.monotonic()
        self._phase_started = time.monotonic()

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.colour else text

    def format(self, record: logging.LogRecord) -> str:
        clock = time.strftime("%H:%M:%S")
        message = record.getMessage()
        if getattr(record, STAGE_FLAG, False):
            now = time.monotonic()
            previous = now - self._phase_started
            self._phase_started = now
            total = now - self._run_started
            timing = self._paint(
                f"(上一步 {previous:5.1f}s / 累计 {total:5.1f}s)", "90"
            )
            head = self._paint(f"▸ {clock}", "1;36")
            return f"\n{head}  {self._paint(message, '1')}  {timing}"
        if record.levelno >= logging.ERROR:
            return f"  {self._paint('✖', '1;31')} {clock}  {message}"
        if record.levelno >= logging.WARNING:
            return f"  {self._paint('⚠', '1;33')} {clock}  {message}"
        return f"    {self._paint(clock, '90')}  {message}"


@dataclass
class PhaseRecord:
    name: str
    seconds: float


@dataclass
class RunTimeline:
    """Where the wall clock actually went, so slowness is attributable."""

    phases: list[PhaseRecord] = field(default_factory=list)
    _started: float = field(default_factory=time.monotonic)
    _current: str | None = None
    _current_started: float = field(default_factory=time.monotonic)

    def mark(self, name: str) -> None:
        now = time.monotonic()
        if self._current is not None:
            self.phases.append(
                PhaseRecord(self._current, now - self._current_started)
            )
        self._current = name
        self._current_started = now

    def finish(self) -> None:
        if self._current is not None:
            self.phases.append(
                PhaseRecord(
                    self._current, time.monotonic() - self._current_started
                )
            )
            self._current = None

    def render(self, *, slowest: int = 6) -> str:
        self.finish()
        if not self.phases:
            return ""
        total = sum(phase.seconds for phase in self.phases)
        ranked = sorted(
            self.phases, key=lambda phase: phase.seconds, reverse=True
        )[:slowest]
        lines = [
            "",
            f"耗时分布（总计 {total:.1f}s，最慢 {len(ranked)} 步）:",
        ]
        for phase in ranked:
            share = phase.seconds / total if total > 0 else 0.0
            bar = "█" * max(1, round(share * 28))
            lines.append(
                f"  {phase.seconds:6.1f}s  {share:4.0%}  {bar}  {phase.name}"
            )
        return "\n".join(lines)


class ProgressReporter:
    """Live progress for a long loop, with a measured ETA.

    On a terminal this repaints one line in place so a 150-step motion does
    not scroll the phase structure away.  Without a terminal it falls back to
    throttled log records, so a piped/redirected run still shows progress
    instead of a silent gap.
    """

    def __init__(
        self,
        label: str,
        total: int,
        *,
        logger: logging.Logger,
        stream: TextIO | None = None,
        min_interval_s: float = 1.0,
    ):
        self.label = label
        self.total = max(1, int(total))
        self.logger = logger
        self.stream = sys.stderr if stream is None else stream
        self.min_interval_s = float(min_interval_s)
        self.interactive = _supports_colour(self.stream)
        self._started = time.monotonic()
        self._last_emit = 0.0
        self._painted = False

    def update(self, done: int, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_emit < self.min_interval_s:
            return
        self._last_emit = now
        elapsed = now - self._started
        done = max(0, min(int(done), self.total))
        share = done / self.total
        eta = (elapsed / done) * (self.total - done) if done > 0 else 0.0
        text = (
            f"{self.label} {done}/{self.total} ({share:3.0%})"
            f"  已用 {elapsed:.0f}s  预计还需 {eta:.0f}s"
        )
        if self.interactive:
            self._painted = True
            self.stream.write(f"\r    \033[90m{text}\033[0m\033[K")
            self.stream.flush()
        else:
            self.logger.info(text)

    def close(self, summary: str | None = None) -> None:
        elapsed = time.monotonic() - self._started
        if self.interactive and self._painted:
            self.stream.write("\r\033[K")
            self.stream.flush()
        self.logger.info(
            "%s 完成：%d 点，用时 %.1fs%s",
            self.label,
            self.total,
            elapsed,
            f"（{summary}）" if summary else "",
        )


def install(*, latest_log: Path | str) -> RunTimeline:
    """Route full detail to files and a readable summary to the terminal."""
    # DEBUG at the root so per-frame evidence still reaches the files; the
    # console handler filters itself back up to INFO.
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        ConsoleFormatter(colour=_supports_colour(sys.stdout))
    )
    root.addHandler(console)

    # Only latest.log is owned here.  RunOrchestrator attaches (and detaches) its
    # own per-run run.log handler; adding a second one would double every
    # line in the run's own evidence file.
    path = Path(latest_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    root.addHandler(handler)

    # Third-party INFO chatter belongs in the file, not in the operator's view.
    for noisy in ("ultralytics", "matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return RunTimeline()
