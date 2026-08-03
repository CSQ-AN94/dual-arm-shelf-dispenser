#!/usr/bin/env python3
"""Offline benchmark summarizer for bottle-grasp run artefacts.

This script is deliberately read-only: it never imports the robot stack and
never opens a camera or controller connection.  Point it at one or more copied
``outputs/`` directories to obtain a CSV/JSON baseline from
``run.log``, ``task_journal.jsonl`` and MoveIt plan/validation artefacts.

Wall-clock buckets inferred from log markers are best-effort.  The raw event
timestamps, task phases, and MoveIt-reported planning times remain separate in
the output so inferred time is never confused with planner CPU time.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence


LOG_LINE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    r"(?:[,.]\d{1,6})?)\s+(?P<level>[A-Z]+)\s+(?P<message>.*)$"
)

# The 2026-07-18 controller sent every safety-interpolated joint sample, while
# current controller-continuous runs send only MoveIt control points after the
# same dense safety review.  Keep both values: physical commands and reviewed
# states answer different operational questions.
LEGACY_TRAJECTORY_START = re.compile(
    r"SDK\s*执行\s*MoveIt\s*轨迹\s*[:：]\s*"
    r"(?P<dense_points>\d+)\s*个(?:密集关节(?:点)?|密集点)"
)
CONTROL_POINT_TRAJECTORY_START = re.compile(
    r"SDK\s*执行\s*MoveIt\s*轨迹\s*[:：]\s*"
    r"(?P<control_points>\d+)\s*个控制点"
    r"(?:\s*[（(]\s*安全复核\s*(?P<dense_points>\d+)\s*个密集点\s*[）)])?"
)
TRAJECTORY_END = re.compile(
    r"轨迹执行\s*完成\s*[:：]\s*"
    r"(?:(?P<points>\d+)\s*点\s*[,，]\s*)?"
    r"用时\s*(?P<seconds>\d+(?:\.\d+)?)\s*s"
)

LOCALIZATION_START = re.compile(r"连续采集\s*\d+\s*帧")
LOCALIZATION_END = re.compile(
    r"共识帧\s*\d+\s*/\s*\d+|"
    r"检测/深度稳定帧不足|多帧定位(?:没有稳定共识|离群过滤后不足|共识门槛配置无效)|"
    r"多帧三维位置过散|相机内参不可用|"
    r"相机.*(?:失败|缺失|不可用|无新鲜帧)|无画面|深度.*不可用"
)
SAFE_PLANNING_START = re.compile(r"安全规划尝试")
SAFE_PLANNING_END = re.compile(
    r"安全轨迹确定|在\s*\d+\s*次安全规划后仍无可执行轨迹"
)
CANDIDATE_READY = re.compile(r"(?:生成.*(?:观察位)?候选|候选.*端点通过)")
CANDIDATE_WINDOW_START = re.compile(
    r"任务/(?:move_to_observation|move_to_observation_staging)|"
    r"(?:构建|重建)障碍场景"
)
MOVEIT_PLAN_SUCCESS = re.compile(
    r"MoveIt2?\s*规划\s+.*?成功\s*[:：]\s*\d+\s*点\s*[,，]\s*"
    r"(?P<seconds>\d+(?:\.\d+)?)\s*s"
)
COLLISION_REVIEW_END = re.compile(
    r"轨迹\s*MoveIt\s*碰撞复核通过|轨迹独立复核拒绝|安全轨迹确定"
)
COLLISION_REVIEW_SUCCESS = re.compile(
    r"轨迹\s*MoveIt\s*碰撞复核通过\s*[:：]\s*(?P<states>\d+)\s*个状态"
)
SCENE_PHASE_START = re.compile(r"任务/(?:scene_sync|output_scene_sync)")
SCENE_BUILD_COMPLETE = re.compile(r"(?:构建|重建)障碍场景")
EXECUTION_ABORT = re.compile(
    r"安全中止|未处理异常|轨迹(?:执行)?(?:失败|.*中止)|"
    r"外部停止指令|STOP(?:/Ctrl\+C)?"
)

CANDIDATE_ENDPOINT_COUNT = re.compile(r"端点通过\s*(?P<count>\d+)\s*个")
CANDIDATE_IK_VIABLE_COUNT = re.compile(
    r"(?:抓取预演可行|逆解(?:/围栏)?可行)[^\d]*(?P<count>\d+)\s*个"
)


FAILURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "operator_stop",
        re.compile(r"(?:外部停止指令|用户.*中止|STOP(?:/Ctrl\+C)?|Ctrl\+C)"),
    ),
    (
        "grasp_success_misjudged",
        re.compile(r"抓取.*误判|真实成功.*误判|pos=402|夹稳.*误判"),
    ),
    (
        "controller_or_tracking",
        re.compile(
            r"(?:rc=-?\d+|API2 -6|控制器错误|关节通信|0x[0-9A-Fa-f]+|"
            r"跟踪(?:反馈)?偏差|到位偏差)"
        ),
    ),
    (
        "singularity_or_joint_limit",
        re.compile(r"奇异|限位|J[1-7].*过近|关节跳变"),
    ),
    (
        "collision_disagreement",
        re.compile(
            r"MoveIt\s*(?:密集复核)?\s*=\s*通过.*(?:SDK\s*)?围栏\s*=\s*拒绝|"
            r"(?:SDK\s*)?围栏\s*=\s*拒绝.*MoveIt\s*(?:密集复核)?\s*=\s*通过|"
            r"MoveIt 说.*碰|深入.*clearance"
        ),
    ),
    (
        "collision_or_fence_rejection",
        re.compile(r"碰撞|围栏|禁入|轨迹.*离开允许区|场景.*过期"),
    ),
    (
        "moveit_planning",
        re.compile(r"MoveIt.*(?:超时|失败|error=99999)|规划服务超时"),
    ),
    (
        "reachability_or_ik",
        re.compile(r"逆解|\bIK\b|不可达|无可执行轨迹"),
    ),
    (
        "perception_or_camera",
        re.compile(
            r"检测/深度|未检测|定位.*(?:不足|失败|跳变|过散)|"
            r"相机.*(?:失败|缺失|不可用|无新鲜帧)|无画面|深度.*不可用"
        ),
    ),
    (
        "software_or_integration",
        re.compile(r"未处理异常|ValueError|Traceback|TCP.*失败|服务启动"),
    ),
)


@dataclass(frozen=True)
class LogEvent:
    timestamp: datetime
    level: str
    message: str


@dataclass
class RunSummary:
    run: str
    status: str
    phase: str
    object_state: str
    failure_category: str
    duration_s: float | None
    localization_s: float
    localization_count: int
    scene_sync_s: float
    scene_sync_count: int
    candidate_and_ik_wall_s: float
    candidate_endpoint_count: int
    candidate_ik_viable_count: int
    candidate_and_moveit_wall_s: float
    moveit_and_collision_wall_s: float
    moveit_attempt_count: int
    collision_review_wall_s: float
    execution_s: float
    execution_control_points: int
    dense_execution_points: int
    moveit_reported_planning_s: float
    moveit_plan_count: int
    collision_validation_count: int
    collision_checked_states: int
    source_dir: str


def _parse_timestamp(raw: str) -> datetime:
    # ``logging.Formatter`` normally uses a comma and milliseconds, but copied
    # field logs also exist with a dot separator or no fractional seconds.
    return datetime.fromisoformat(raw.replace(",", "."))


def read_events(path: Path) -> list[LogEvent]:
    events: list[LogEvent] = []
    if not path.is_file():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOG_LINE.match(line)
        if match is None:
            continue
        try:
            timestamp = _parse_timestamp(match.group("timestamp"))
        except ValueError:
            # A damaged line should not make an otherwise useful copied log
            # impossible to summarize.
            continue
        events.append(
            LogEvent(
                timestamp=timestamp,
                level=match.group("level"),
                message=match.group("message"),
            )
        )
    return events


def _safe_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _journal_events(path: Path) -> list[dict]:
    result: list[dict] = []
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("timestamp"):
            result.append(item)
    return result


def _seconds_between(first: datetime, second: datetime) -> float:
    return max(0.0, (second - first).total_seconds())


def _journal_timestamp(value: object) -> datetime:
    """Parse journal timestamps into a naive UTC value.

    Run logs are local naive timestamps, whereas task journals use UTC ISO
    timestamps.  Their durations are only computed within each source, but
    normalizing journal values here also accepts both ``Z`` and explicit UTC
    offsets without risking naive/aware subtraction errors.
    """
    raw = str(value)
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _bounded_intervals(
    events: Sequence[LogEvent],
    starts: re.Pattern[str],
    ends: re.Pattern[str],
    *,
    close_on_restart: bool = False,
) -> tuple[float, int]:
    """Sum non-overlapping marker intervals and return seconds/count."""
    opened: datetime | None = None
    total = 0.0
    count = 0
    for event in events:
        if starts.search(event.message):
            if opened is None:
                opened = event.timestamp
            elif close_on_restart:
                total += _seconds_between(opened, event.timestamp)
                count += 1
                opened = event.timestamp
            continue
        if opened is not None and ends.search(event.message):
            total += _seconds_between(opened, event.timestamp)
            count += 1
            opened = None
    return total, count


def _phase_durations(journal: Sequence[dict]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for current, following in zip(journal, journal[1:]):
        try:
            start = _journal_timestamp(current["timestamp"])
            end = _journal_timestamp(following["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        phase = str(current.get("phase", "unknown"))
        durations[phase] = durations.get(phase, 0.0) + _seconds_between(start, end)
    return durations


def _phase_event_count(journal: Sequence[dict], phases: set[str]) -> int:
    return sum(1 for event in journal if str(event.get("phase")) in phases)


def _intervals_from_latest_start(
    events: Sequence[LogEvent],
    starts: re.Pattern[str],
    ends: re.Pattern[str],
) -> tuple[float, int]:
    """Measure end markers from the most recent relevant preceding start.

    Candidate generation logs only a completion stage, while task mode logs
    its surrounding phase at the beginning.  Replacing an unmatched start is
    intentional: a newer operation supersedes an unfinished earlier one.
    """
    opened: datetime | None = None
    total = 0.0
    count = 0
    for event in events:
        if starts.search(event.message):
            opened = event.timestamp
            continue
        if opened is not None and ends.search(event.message):
            total += _seconds_between(opened, event.timestamp)
            count += 1
            opened = None
    return total, count


def _trajectory_start(message: str) -> tuple[int, int] | None:
    """Return physical control points and independently reviewed dense points."""
    current = CONTROL_POINT_TRAJECTORY_START.search(message)
    if current is not None:
        control_points = int(current.group("control_points"))
        dense_raw = current.group("dense_points")
        # A partially copied current-format line is still meaningful.  In
        # that case only the command count is known, so do not invent a larger
        # safety-review count.
        dense_points = int(dense_raw) if dense_raw is not None else control_points
        return control_points, dense_points
    legacy = LEGACY_TRAJECTORY_START.search(message)
    if legacy is not None:
        dense_points = int(legacy.group("dense_points"))
        return dense_points, dense_points
    return None


def _execution_metrics(events: Sequence[LogEvent]) -> tuple[float, int, int]:
    """Extract execution time plus command/review point counts from run logs."""
    execution_s = 0.0
    control_points = 0
    dense_points = 0
    pending_execution: datetime | None = None
    for event in events:
        start = _trajectory_start(event.message)
        if start is not None:
            # A new SDK trajectory after an unclosed prior one means the old
            # run was interrupted.  Preserve only its observed wall time;
            # never extrapolate to the end of a potentially truncated log.
            if pending_execution is not None:
                execution_s += _seconds_between(pending_execution, event.timestamp)
            pending_execution = event.timestamp
            control_points += start[0]
            dense_points += start[1]
            continue

        end_match = TRAJECTORY_END.search(event.message)
        if end_match is not None:
            if pending_execution is not None:
                execution_s += float(end_match.group("seconds"))
                pending_execution = None
            continue

        if pending_execution is not None and (
            event.level in {"ERROR", "CRITICAL"}
            or EXECUTION_ABORT.search(event.message)
        ):
            execution_s += _seconds_between(pending_execution, event.timestamp)
            pending_execution = None
    return execution_s, control_points, dense_points


def _candidate_metrics(events: Sequence[LogEvent]) -> tuple[int, int]:
    endpoints = 0
    viable = 0
    for event in events:
        if not CANDIDATE_READY.search(event.message):
            continue
        endpoint_match = CANDIDATE_ENDPOINT_COUNT.search(event.message)
        if endpoint_match is not None:
            endpoints += int(endpoint_match.group("count"))
        viable_match = CANDIDATE_IK_VIABLE_COUNT.search(event.message)
        if viable_match is not None:
            viable += int(viable_match.group("count"))
    return endpoints, viable


def _failure_category(
    text: str,
    status: str,
    terminal_text: str | Sequence[str] = "",
) -> str:
    if status.strip().lower() in {"done", "success", "succeeded", "completed"}:
        return "none"
    # A successful retry can leave earlier rejection text in the log.  For a
    # failed run, classify the durable terminal error (or latest error line)
    # first; only then fall back to the complete forensic log.
    if isinstance(terminal_text, str):
        terminal_evidence = (terminal_text,)
    else:
        terminal_evidence = tuple(terminal_text)
    for evidence in (*terminal_evidence, text):
        if not evidence:
            continue
        for label, pattern in FAILURE_PATTERNS:
            if pattern.search(evidence):
                return label
    return "unknown_or_insufficient_evidence"


def _moveit_metrics(run_dir: Path) -> tuple[float, int, int, int]:
    planning_s = 0.0
    plans = 0
    validations = 0
    checked_states = 0
    for path in run_dir.glob("*_plan.json"):
        payload = _safe_json(path)
        if not payload:
            continue
        plans += 1
        value = next(
            (
                payload.get(key)
                for key in ("planning_time", "planning_time_s", "planning_seconds")
                if isinstance(payload.get(key), (int, float))
                and not isinstance(payload.get(key), bool)
            ),
            None,
        )
        if value is not None:
            planning_s += float(value)
    for path in run_dir.glob("*_validation.json"):
        payload = _safe_json(path)
        if not payload:
            continue
        validations += 1
        value = next(
            (
                payload.get(key)
                for key in ("checked_states", "checked_state_count", "states_checked")
                if isinstance(payload.get(key), int)
                and not isinstance(payload.get(key), bool)
            ),
            None,
        )
        if value is not None:
            checked_states += int(value)
    return planning_s, plans, validations, checked_states


def _moveit_log_metrics(events: Sequence[LogEvent]) -> tuple[float, int, int, int]:
    """Fallback when copied logs have no JSON artefacts beside them."""
    planning_s = 0.0
    plans = 0
    validations = 0
    checked_states = 0
    for event in events:
        plan_match = MOVEIT_PLAN_SUCCESS.search(event.message)
        if plan_match is not None:
            plans += 1
            planning_s += float(plan_match.group("seconds"))
        validation_match = COLLISION_REVIEW_SUCCESS.search(event.message)
        if validation_match is not None:
            validations += 1
            checked_states += int(validation_match.group("states"))
    return planning_s, plans, validations, checked_states


def _terminal_evidence(
    events: Sequence[LogEvent],
    journal: Sequence[dict],
    result: dict,
) -> tuple[str, ...]:
    parts: list[str] = []
    if result.get("error"):
        parts.append(str(result["error"]))
    if journal:
        message = journal[-1].get("message")
        if message:
            parts.append(str(message))
    for event in reversed(events):
        if event.level in {"ERROR", "CRITICAL"}:
            parts.append(event.message)
            break
    return tuple(parts)


def _journal_duration(journal: Sequence[dict]) -> float | None:
    timestamps: list[datetime] = []
    for event in journal:
        try:
            timestamps.append(_journal_timestamp(event["timestamp"]))
        except (KeyError, TypeError, ValueError):
            continue
    if len(timestamps) < 2:
        return None
    return _seconds_between(timestamps[0], timestamps[-1])


def summarize_run(run_dir: Path) -> RunSummary:
    events = read_events(run_dir / "run.log")
    journal = _journal_events(run_dir / "task_journal.jsonl")
    result = _safe_json(run_dir / "task_result.json")
    phase_times = _phase_durations(journal)

    localization_s, localization_count = _bounded_intervals(
        events,
        LOCALIZATION_START,
        LOCALIZATION_END,
        close_on_restart=True,
    )
    moveit_and_collision_wall_s, _ = _bounded_intervals(
        events,
        SAFE_PLANNING_START,
        SAFE_PLANNING_END,
        close_on_restart=True,
    )
    candidate_and_ik_wall_s, _ = _intervals_from_latest_start(
        events,
        CANDIDATE_WINDOW_START,
        CANDIDATE_READY,
    )
    collision_review_wall_s, _ = _bounded_intervals(
        events,
        MOVEIT_PLAN_SUCCESS,
        COLLISION_REVIEW_END,
        close_on_restart=True,
    )
    execution_s, execution_control_points, dense_points = _execution_metrics(events)
    candidate_endpoints, candidate_ik_viable = _candidate_metrics(events)
    moveit_attempt_count = sum(
        1 for event in events if SAFE_PLANNING_START.search(event.message)
    )

    moveit_s, plan_count, validation_count, checked_states = _moveit_metrics(
        run_dir
    )
    (
        logged_moveit_s,
        logged_plan_count,
        logged_validation_count,
        logged_checked_states,
    ) = _moveit_log_metrics(events)
    # Copied evidence often keeps only one side of the plan/validation JSON
    # pair.  Fall back independently so a surviving plan file does not hide
    # collision-review evidence still present in run.log (or vice versa).
    if plan_count == 0:
        moveit_s = logged_moveit_s
        plan_count = logged_plan_count
    elif moveit_s == 0.0 and logged_moveit_s > 0.0:
        moveit_s = logged_moveit_s
    if validation_count == 0:
        validation_count = logged_validation_count
        checked_states = logged_checked_states
    elif checked_states == 0 and logged_checked_states > 0:
        checked_states = logged_checked_states

    scene_phases = {"scene_sync", "output_scene_sync"}
    scene_sync_s = sum(phase_times.get(phase, 0.0) for phase in scene_phases)
    scene_sync_count = _phase_event_count(journal, scene_phases)
    if scene_sync_count == 0:
        # Legacy demo runs did not create a task journal.  They can still be
        # measured when the phase stage was written to the log.
        scene_sync_s, scene_sync_count = _bounded_intervals(
            events,
            SCENE_PHASE_START,
            SCENE_BUILD_COMPLETE,
            close_on_restart=True,
        )

    if events:
        duration_s: float | None = _seconds_between(
            events[0].timestamp, events[-1].timestamp
        )
    else:
        duration_s = _journal_duration(journal)

    status = str(result.get("status") or "unknown")
    phase = str(
        result.get("phase")
        or (journal[-1].get("phase") if journal else "unknown")
    )
    object_state = str(
        result.get("object_state")
        or (journal[-1].get("object_state") if journal else "unknown")
    )
    text = "\n".join(event.message for event in events)
    if result.get("error"):
        text += "\n" + str(result["error"])

    return RunSummary(
        run=run_dir.name,
        status=status,
        phase=phase,
        object_state=object_state,
        failure_category=_failure_category(
            text,
            status,
            _terminal_evidence(events, journal, result),
        ),
        duration_s=duration_s,
        localization_s=round(localization_s, 3),
        localization_count=localization_count,
        scene_sync_s=round(scene_sync_s, 3),
        scene_sync_count=scene_sync_count,
        candidate_and_ik_wall_s=round(candidate_and_ik_wall_s, 3),
        candidate_endpoint_count=candidate_endpoints,
        candidate_ik_viable_count=candidate_ik_viable,
        # Retain the original output column as a stable alias for callers
        # that consumed this script before MoveIt/collision timing split out.
        candidate_and_moveit_wall_s=round(moveit_and_collision_wall_s, 3),
        moveit_and_collision_wall_s=round(moveit_and_collision_wall_s, 3),
        moveit_attempt_count=moveit_attempt_count,
        collision_review_wall_s=round(collision_review_wall_s, 3),
        execution_s=round(execution_s, 3),
        execution_control_points=execution_control_points,
        dense_execution_points=dense_points,
        moveit_reported_planning_s=round(moveit_s, 6),
        moveit_plan_count=plan_count,
        collision_validation_count=validation_count,
        collision_checked_states=checked_states,
        source_dir=str(run_dir.resolve()),
    )


def discover_runs(roots: Iterable[Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] if (root / "run.log").is_file() else root.iterdir()
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            if not any(
                (candidate / name).exists()
                for name in ("run.log", "task_journal.jsonl", "task_result.json")
            ):
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield candidate


def _render_csv(summaries: Sequence[RunSummary]) -> None:
    if not summaries:
        return
    writer = csv.DictWriter(sys.stdout, fieldnames=list(asdict(summaries[0])))
    writer.writeheader()
    writer.writerows(asdict(summary) for summary in summaries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize copied bottle-grasp run artefacts without hardware access"
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="run directory or parent outputs/shelf_dispenser directory",
    )
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summaries = [summarize_run(path) for path in discover_runs(args.roots)]
    summaries.sort(key=lambda item: item.run)
    if args.format == "csv":
        _render_csv(summaries)
    else:
        print(
            json.dumps(
                [asdict(summary) for summary in summaries],
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if summaries else 2


if __name__ == "__main__":
    raise SystemExit(main())
