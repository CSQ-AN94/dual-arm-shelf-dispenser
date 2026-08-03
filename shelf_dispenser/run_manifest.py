"""Immutable, machine-readable provenance for a bottle-grasp run.

The manifest deliberately captures only local files, parsed command-line
arguments and a small allow-list of execution environment variables.  It
does not create a robot session, open a camera, or invoke a shell.  Call
``write_run_manifest`` immediately after the run evidence directory is
created and before any hardware-facing setup.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

try:
    from .core import SafetyAbort
except ImportError:  # runpy.run_path() for launcher source provenance
    class SafetyAbort(RuntimeError):
        """Standalone fallback; package execution imports the real type."""



MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_SCHEMA_VERSION = 1

# Keep this list narrow: command provenance is useful, but a run manifest
# must never accidentally copy API credentials or unrelated user variables.
_COMMAND_ARGUMENTS = (
    "task_mode",
    "execute",
    "plan_only",
    "stop_after_observation",
    "confirm_before_grasp",
    "safety_profile",
    "delivery_safety_profile",
    "dispense",
    "target_product",
    "visual_servo_mode",
    "visual_servo",
    "commissioning_speed",
    "port",
)

_ENVIRONMENT_VARIABLES = (
    # The public launcher computes these in its source checkout before rsync.
    # The robot-side checkout may intentionally lack .git, so a complete set
    # is authoritative over local git plumbing below.
    "BOTTLE_GRASP_SOURCE_GIT_SHA",
    "BOTTLE_GRASP_SOURCE_DIRTY",
    "BOTTLE_GRASP_SOURCE_DIRTY_DIGEST",
    "BOTTLE_GRASP_SOURCE_DIRTY_DIGEST_ALGORITHM",
    # This is the actual remote execution switch consumed by RobotSession.
    "BOTTLE_GRASP_CONTINUOUS_TRAJECTORY",
    # These are retained if a caller invokes the Python entrypoint directly
    # with the launcher-facing names rather than the normalized CLI flags.
    "BOTTLE_GRASP_TRAJECTORY_MODE",
    "VISUAL_MODE",
    "VISUAL_SERVO",
    "COMMISSIONING_SPEED",
)


# This alias is evaluated at module import time (unlike postponed function
# annotations below), so keep it compatible with the system Python 3.9 used
# by the Mac launcher to capture source provenance before touching hardware.
GitRunner = Callable[[Path, Sequence[str]], Optional[bytes]]


def _json_value(value: Any) -> Any:
    """Convert small config/argument values into canonical JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        # ``allow_nan=False`` below gives a meaningful deterministic manifest
        # instead of serialising platform-specific NaN spellings.
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist())
    # CalibrationConfig is a dataclass in production; using __dict__ keeps
    # this helper independent from the configuration package and test doubles.
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return _json_value(attributes)
    return str(value)


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        _json_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _file_identity(path: Path) -> dict[str, Any]:
    """Return a content identity without embedding artifact contents."""
    record: dict[str, Any] = {"path": str(path), "sha256": None}
    try:
        stat = path.stat()
    except FileNotFoundError:
        record["state"] = "missing"
        return record
    except OSError as exc:
        record["state"] = "unreadable"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    if not path.is_file():
        record["state"] = "not_a_regular_file"
        return record

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        record["state"] = "unreadable"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record.update(
        state="present",
        sha256=digest.hexdigest(),
        size_bytes=stat.st_size,
    )
    return record


def _default_git_runner(project_root: Path, args: Sequence[str]) -> bytes | None:
    """Run local git plumbing only; never use a shell or a network remote."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    return completed.stdout if completed.returncode == 0 else None


def _untracked_file_digests(project_root: Path, status: bytes) -> list[dict[str, str]]:
    """Hash untracked content too, so a path-only status cannot collide."""
    entries: list[dict[str, str]] = []
    for raw_entry in status.split(b"\0"):
        if not raw_entry.startswith(b"?? "):
            continue
        relative = Path(os.fsdecode(raw_entry[3:]))
        # Git status paths are repository-relative.  Refuse a malformed test
        # double or repository entry rather than walking outside the project.
        # ``abspath`` normalizes ``..`` without resolving a symlink.  That
        # lets us hash a repository-local symlink itself rather than following
        # it to an unrelated file outside the checkout.
        candidate = Path(os.path.abspath(project_root / relative))
        try:
            candidate.relative_to(project_root)
        except ValueError:
            continue
        try:
            mode = candidate.lstat().st_mode
            if os.path.islink(candidate):
                content = b"symlink\0" + os.fsencode(os.readlink(candidate))
            elif os.path.isfile(candidate):
                content = candidate.read_bytes()
            else:
                content = f"unsupported-mode:{mode:o}".encode("ascii")
        except OSError as exc:
            content = f"unreadable:{type(exc).__name__}".encode("ascii")
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256_bytes(content),
            }
        )
    return sorted(entries, key=lambda item: item["path"])


def collect_git_provenance(
    project_root: str | Path,
    *,
    git_runner: GitRunner | None = None,
) -> dict[str, Any]:
    """Capture the checked-out SHA plus a deterministic dirty-tree digest."""
    root = Path(project_root).resolve()
    runner = git_runner or _default_git_runner
    commit = runner(root, ("rev-parse", "--verify", "HEAD"))
    status = runner(
        root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
    )
    diff = runner(root, ("diff", "--binary", "--no-ext-diff", "HEAD"))

    if commit is None or status is None or diff is None:
        return {
            "commit_sha": None,
            "dirty": None,
            "dirty_digest": None,
            "state": "unavailable",
        }

    untracked = _untracked_file_digests(root, status)
    dirty = bool(status)
    dirty_payload = {
        "algorithm": "git-diff-head-plus-untracked-content-v1",
        "status_sha256": _sha256_bytes(status),
        "tracked_diff_sha256": _sha256_bytes(diff),
        "untracked": untracked,
    }
    return {
        "commit_sha": commit.decode("ascii", "replace").strip(),
        "dirty": dirty,
        "dirty_digest": _sha256_json(dirty_payload),
        "dirty_digest_algorithm": dirty_payload["algorithm"],
        "untracked_file_count": len(untracked),
        "state": "available",
    }


def _launcher_source_git_provenance(
    environment: Mapping[str, str],
) -> dict[str, Any] | None:
    """Return source-checkout provenance when the launcher supplied all of it.

    A partial override is worse than the local fallback because it can pair a
    source SHA with an unrelated robot-side dirty digest.  Therefore this
    accepts only a complete, explicitly boolean set of values.
    """
    sha = environment.get("BOTTLE_GRASP_SOURCE_GIT_SHA")
    dirty = environment.get("BOTTLE_GRASP_SOURCE_DIRTY")
    dirty_digest = environment.get("BOTTLE_GRASP_SOURCE_DIRTY_DIGEST")
    algorithm = environment.get(
        "BOTTLE_GRASP_SOURCE_DIRTY_DIGEST_ALGORITHM"
    )
    if not all((sha, dirty, dirty_digest, algorithm)) or dirty not in {"0", "1"}:
        return None
    return {
        "commit_sha": sha,
        "dirty": dirty == "1",
        "dirty_digest": dirty_digest,
        "dirty_digest_algorithm": algorithm,
        "untracked_file_count": None,
        "state": "available",
        "provenance": "launcher_source",
    }


def _config_path(args: object, project_root: Path) -> Path:
    configured = getattr(args, "config", None)
    if configured:
        return Path(configured).expanduser().resolve()
    return (project_root / "config.yaml").resolve()


def _safety_config_path(
    args: object, project_root: Path
) -> Path | None:
    configured = getattr(args, "safety_config", None)
    if not configured:
        return None
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _raw_config(path: Path) -> Mapping[str, Any] | None:
    """Best-effort YAML read for lightweight/non-AppConfig callers."""
    try:
        import yaml

        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except (ImportError, OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _effective_calibration(
    config: object | None, raw_config: Mapping[str, Any] | None
) -> tuple[Any, str]:
    calibration = getattr(config, "calibration", None) if config is not None else None
    if calibration is not None:
        return calibration, "effective_config"
    if raw_config is not None:
        return raw_config.get("calibration"), "config_file"
    return None, "unavailable"


def _model_path(
    config: object | None,
    raw_config: Mapping[str, Any] | None,
    config_path: Path,
) -> tuple[Path | None, str | None]:
    vision = getattr(config, "vision", None) if config is not None else None
    configured = getattr(vision, "model_path", None) if vision is not None else None
    if configured is None and raw_config is not None:
        raw_vision = raw_config.get("vision")
        if isinstance(raw_vision, Mapping):
            configured = raw_vision.get("model_path")
    if not configured:
        return None, None
    model_path = Path(str(configured)).expanduser()
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    return model_path.resolve(), str(configured)


def _model_asset_contract(
    project_root: Path, model_path: Path | None
) -> dict[str, Any]:
    """Load the pure asset inspector without breaking launcher ``run_path``.

    ``scripts/run_task.sh`` loads this file by path on a development
    Mac solely to collect Git provenance.  Keep that route independent from
    package imports and their robot/vision dependencies.
    """
    if __package__:
        from .model_assets import inspect_model_asset_contract

        return inspect_model_asset_contract(project_root, model_path)
    spec = importlib.util.spec_from_file_location(
        "shelf_dispenser_model_assets_standalone",
        Path(__file__).with_name("model_assets.py"),
    )
    if spec is None or spec.loader is None:
        return {
            "schema_version": 1,
            "assets": [],
            "optional_unavailable": [],
            "state": "inspector_unavailable",
        }
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inspect_model_asset_contract(project_root, model_path)


def _resolved_visual_mode(args: object) -> str:
    mode = getattr(args, "visual_servo_mode", None)
    if mode is None:
        return "active" if bool(getattr(args, "visual_servo", False)) else "off"
    return str(mode)


def _speed_snapshot(params: object | None) -> dict[str, int | None]:
    return {
        "transit_percent": getattr(params, "transit_speed", None),
        "travel_percent": getattr(params, "travel_speed", None),
        "final_percent": getattr(params, "final_speed", None),
        "gripper_percent": getattr(params, "gripper_speed", None),
    }


def _effective_safety_profile(
    safety_config_path: Path | None, profile_name: object
) -> dict[str, Any]:
    """Capture the selected safety profile, not only its containing file.

    The whole-file digest remains useful for tamper detection, but it cannot
    tell an investigator which B/C behaviour was selected when one JSON file
    contains many profiles.  This helper is pure local file parsing and is
    intentionally called before any hardware-facing initialization.
    """
    name = None if profile_name is None else str(profile_name)
    record: dict[str, Any] = {
        "name": name,
        "state": (
            "not_configured"
            if safety_config_path is None or name is None
            else "unavailable"
        ),
        "sha256": None,
        "config": None,
    }
    if safety_config_path is None or name is None:
        return record
    try:
        payload = json.loads(safety_config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        record["state"] = "missing"
        return record
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        record["state"] = "unreadable"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    profiles = payload.get("profiles") if isinstance(payload, Mapping) else None
    selected = profiles.get(name) if isinstance(profiles, Mapping) else None
    if not isinstance(selected, Mapping):
        record["state"] = "missing_profile"
        return record
    effective = _json_value(selected)
    record.update(
        state="present",
        sha256=_sha256_json(effective),
        config=effective,
    )
    return record


def manifest_profile_expectations(
    manifest_path: str | Path,
    *,
    args: object,
    project_root: str | Path,
) -> dict[str, str | None] | None:
    """Return frozen profile digests for the manifest-backed execution path.

    The returned hashes are passed into the profile loader, which compares
    them with the exact JSON object it parses. This closes the gap between
    writing a manifest and later reopening the profile path during
    initialization. A missing manifest is tolerated for offline embedders;
    a present but malformed or stale manifest fails closed.
    """
    path = Path(manifest_path)
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SafetyAbort(
            f"run manifest 无法读取，拒绝加载 safety profile: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise SafetyAbort("run manifest 根对象无效，拒绝加载 safety profile")
    profiles = manifest.get("profiles")
    if not isinstance(profiles, Mapping):
        raise SafetyAbort("run manifest 缺少 profiles，拒绝加载 safety profile")

    root = Path(project_root).resolve()
    current_config_path = _safety_config_path(args, root)
    expected_file = profiles.get("safety_config")
    if current_config_path is None:
        if isinstance(expected_file, Mapping) and expected_file.get("state") in {
            "not_configured",
            None,
        }:
            return {"source": None, "delivery": None}
        raise SafetyAbort("run manifest 与当前 safety_config 参数不一致")
    if not isinstance(expected_file, Mapping):
        raise SafetyAbort("run manifest 缺少 safety_config 身份，拒绝继续")
    actual_file = _file_identity(current_config_path)
    if (
        expected_file.get("state") != "present"
        or actual_file.get("state") != "present"
        or expected_file.get("path") != str(current_config_path)
        or expected_file.get("sha256") != actual_file.get("sha256")
    ):
        raise SafetyAbort(
            "run manifest 记录的 safety_config 已改变，拒绝加载不可复现的 profile"
        )

    effective = profiles.get("effective")
    if not isinstance(effective, Mapping):
        raise SafetyAbort("run manifest 缺少 effective profiles，拒绝继续")

    def expected_digest(key: str, argument_name: str) -> str | None:
        selected = getattr(args, argument_name, None)
        record = effective.get(key)
        if selected is None:
            if (
                isinstance(record, Mapping)
                and record.get("state") == "not_configured"
            ):
                return None
            raise SafetyAbort(
                f"run manifest 的 {key} profile 与当前参数不一致"
            )
        if not isinstance(record, Mapping):
            raise SafetyAbort(
                f"run manifest 缺少 {key} profile digest，拒绝继续"
            )
        digest = record.get("sha256")
        if (
            record.get("state") != "present"
            or record.get("name") != str(selected)
            or not isinstance(digest, str)
            or not digest
        ):
            raise SafetyAbort(
                f"run manifest 的 {key} profile 与当前参数不一致"
            )
        return digest

    return {
        "source": expected_digest("source", "safety_profile"),
        "delivery": expected_digest(
            "delivery", "delivery_safety_profile"
        ),
    }


def build_run_manifest(
    *,
    args: object,
    config: object | None,
    project_root: str | Path,
    params: object | None = None,
    environ: Mapping[str, str] | None = None,
    git_runner: GitRunner | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Build provenance without interacting with a camera, SDK, or robot."""
    root = Path(project_root).resolve()
    environment = os.environ if environ is None else environ
    config_path = _config_path(args, root)
    raw_config = _raw_config(config_path)
    calibration, calibration_source = _effective_calibration(config, raw_config)
    model_path, configured_model_path = _model_path(config, raw_config, config_path)
    trajectory_variable = environment.get("BOTTLE_GRASP_CONTINUOUS_TRAJECTORY")
    continuous = trajectory_variable != "0"
    timestamp = created_at or datetime.now(timezone.utc)
    command_arguments = {
        name: _json_value(getattr(args, name, None)) for name in _COMMAND_ARGUMENTS
    }
    command_environment = {
        name: environment.get(name) for name in _ENVIRONMENT_VARIABLES
    }

    safety_config_path = _safety_config_path(args, root)

    model_identity = (
        _file_identity(model_path)
        if model_path is not None
        else {"path": None, "sha256": None, "state": "not_configured"}
    )
    model_identity["configured_path"] = configured_model_path
    model_assets = _model_asset_contract(root, model_path)

    launcher_git = _launcher_source_git_provenance(environment)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": timestamp.astimezone(timezone.utc).isoformat(),
        "git": launcher_git
        or collect_git_provenance(root, git_runner=git_runner),
        "profiles": {
            "source": getattr(args, "safety_profile", None),
            "delivery": getattr(args, "delivery_safety_profile", None),
            "safety_config": (
                _file_identity(safety_config_path)
                if safety_config_path is not None
                else {"path": None, "sha256": None, "state": "not_configured"}
            ),
            # Selected, canonical profile payloads make B shelf-grasp and C
            # body-return settings reproducible even when the same safety
            # config file contains unrelated profiles.
            "effective": {
                "source": _effective_safety_profile(
                    safety_config_path,
                    getattr(args, "safety_profile", None),
                ),
                "delivery": _effective_safety_profile(
                    safety_config_path,
                    getattr(args, "delivery_safety_profile", None),
                ),
            },
        },
        "artifacts": {
            "config": _file_identity(config_path),
            "calibration": {
                "source": calibration_source,
                "sha256": (
                    _sha256_json(calibration) if calibration is not None else None
                ),
            },
            "model": model_identity,
            "model_assets": model_assets,
        },
        "visual_loop": {
            "mode": _resolved_visual_mode(args),
            "legacy_visual_servo_enabled": bool(
                getattr(args, "visual_servo", False)
            ),
            "max_corrections": getattr(
                args, "visual_servo_max_corrections", None
            ),
            "step_mm": getattr(args, "visual_servo_step_mm", None),
            "total_mm": getattr(args, "visual_servo_total_mm", None),
            "convergence_mm": getattr(
                args, "visual_servo_convergence_mm", None
            ),
        },
        "execution": {
            "trajectory_mode": "continuous" if continuous else "blocking",
            "continuous_trajectory": continuous,
            "continuous_trajectory_env": trajectory_variable,
            "commissioning_speed_cap_percent": getattr(
                args, "commissioning_speed", None
            ),
        },
        "speeds": {"effective_percent": _speed_snapshot(params)},
        "stage_entry": {
            "task_mode": getattr(args, "task_mode", None),
            "stop_after_observation": bool(
                getattr(args, "stop_after_observation", False)
            ),
            "confirm_before_grasp": bool(
                getattr(args, "confirm_before_grasp", False)
            ),
        },
        "command_variables": {
            "arguments": command_arguments,
            "environment": command_environment,
        },
    }


def write_run_manifest(
    run_dir: str | Path,
    *,
    args: object,
    config: object | None,
    project_root: str | Path,
    params: object | None = None,
    environ: Mapping[str, str] | None = None,
    git_runner: GitRunner | None = None,
    created_at: datetime | None = None,
) -> Path:
    """Write ``run_manifest.json`` before hardware initialization begins."""
    destination = Path(run_dir) / MANIFEST_FILENAME
    manifest = build_run_manifest(
        args=args,
        config=config,
        project_root=project_root,
        params=params,
        environ=environ,
        git_runner=git_runner,
        created_at=created_at,
    )
    # A run directory has a UUID suffix, so this name is unique in normal
    # operation.  Replace atomically anyway: an interrupted process must not
    # leave a syntactically truncated piece of forensic evidence behind.
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination
