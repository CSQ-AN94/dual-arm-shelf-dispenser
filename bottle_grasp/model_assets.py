"""Fail-closed, offline contract for bottle-detector model assets.

The task must never borrow a model from a dirty checkout or let Ultralytics
download one implicitly.  A tracked lock describes the exact primary model
that an archive must carry.  The historic generic fallback is explicitly
optional: when it has not been separately registered and shipped, callers
receive ``None`` and continue with the verified primary model only.

This module is intentionally filesystem-only.  It imports no camera, robot,
MoveIt, SDK, or ML runtime and is safe to call before any hardware setup.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

LOCK_RELATIVE_PATH = Path("bottle_grasp/model_assets.lock.json")
FALLBACK_RELATIVE_PATH = Path("intelligence/yolo_models/yolo11n.pt")
CONTRACT_ID = "bottle-grasp-yolo-models"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ModelAssetContractError(RuntimeError):
    """Pure-filesystem preflight failure, wrapped by runtime entrypoints."""


def _resolved_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _relative_path(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        # An absolute model path is not permitted by the release lock, but
        # preserve it in the evidence so an operator can correct the config.
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lock_identity(path: Path) -> dict[str, Any]:
    identity: dict[str, Any] = {"path": str(path), "sha256": None}
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        identity["state"] = "missing"
        return identity
    except OSError as exc:
        identity.update(state="unreadable", error=f"{type(exc).__name__}: {exc}")
        return identity
    if not path.is_file():
        identity["state"] = "not_a_regular_file"
        return identity
    try:
        identity.update(
            state="present", size_bytes=size, sha256=_sha256_file(path)
        )
    except OSError as exc:
        identity.update(state="unreadable", error=f"{type(exc).__name__}: {exc}")
    return identity


def _load_lock(lock_path: Path) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    identity = _lock_identity(lock_path)
    if identity["state"] != "present":
        return None, identity
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        identity.update(
            state="invalid_json", error=f"{type(exc).__name__}: {exc}"
        )
        return None, identity
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        identity["state"] = "invalid_schema"
        return None, identity
    if payload.get("contract_id") != CONTRACT_ID:
        identity["state"] = "invalid_contract_id"
        return None, identity
    assets = payload.get("assets")
    if not isinstance(assets, list):
        identity["state"] = "invalid_schema"
        return None, identity
    roles: set[str] = set()
    duplicate_roles: set[str] = set()
    for entry in assets:
        if not isinstance(entry, Mapping):
            continue
        role = entry.get("role")
        if not isinstance(role, str):
            continue
        if role in roles:
            duplicate_roles.add(role)
        roles.add(role)
    if duplicate_roles:
        identity.update(
            state="duplicate_asset_roles",
            duplicate_roles=sorted(duplicate_roles),
        )
        return None, identity
    return payload, identity


def _entries_by_role(payload: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if payload is None:
        return {}
    indexed: dict[str, Mapping[str, Any]] = {}
    for entry in payload.get("assets", []):
        if not isinstance(entry, Mapping):
            continue
        role = entry.get("role")
        if isinstance(role, str) and role not in indexed:
            indexed[role] = entry
    return indexed


def _metadata_issues(entry: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if not isinstance(entry.get("version"), str) or not entry["version"].strip():
        issues.append("version")
    expected_size = entry.get("size_bytes")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        issues.append("size_bytes")
    expected_sha = entry.get("sha256")
    if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(expected_sha):
        issues.append("sha256")
    source_uri = entry.get("source_uri")
    if not isinstance(source_uri, str) or not source_uri.strip():
        issues.append("source_uri")
    return issues


def _asset_record(
    *,
    role: str,
    required: bool,
    project_root: Path,
    configured_path: Path,
    lock_path: Path,
    entry: Mapping[str, Any] | None,
    lock_available: bool,
) -> dict[str, Any]:
    relative_path = _relative_path(project_root, configured_path)
    record: dict[str, Any] = {
        "role": role,
        "required": required,
        "configured_path": relative_path,
        "path": str(configured_path),
        "lock_path": str(lock_path),
        "version": entry.get("version") if entry is not None else None,
        "expected_size_bytes": (
            entry.get("size_bytes") if entry is not None else None
        ),
        "expected_sha256": entry.get("sha256") if entry is not None else None,
        "source_uri": entry.get("source_uri") if entry is not None else None,
    }
    if not lock_available:
        record["state"] = "lock_unavailable"
        return record
    if entry is None:
        record["state"] = "not_registered"
        return record
    if entry.get("relative_path") != relative_path:
        record["state"] = "path_mismatch"
        record["locked_relative_path"] = entry.get("relative_path")
        return record
    metadata_issues = _metadata_issues(entry)
    if metadata_issues:
        record.update(state="not_registered", metadata_issues=metadata_issues)
        return record

    try:
        stat = configured_path.stat()
    except FileNotFoundError:
        record["state"] = "missing"
        return record
    except OSError as exc:
        record.update(state="unreadable", error=f"{type(exc).__name__}: {exc}")
        return record
    if not configured_path.is_file():
        record["state"] = "not_a_regular_file"
        return record

    record["actual_size_bytes"] = stat.st_size
    if stat.st_size != entry["size_bytes"]:
        record["state"] = "size_mismatch"
        return record
    try:
        actual_sha256 = _sha256_file(configured_path)
    except OSError as exc:
        record.update(state="unreadable", error=f"{type(exc).__name__}: {exc}")
        return record
    record["actual_sha256"] = actual_sha256
    if actual_sha256 != entry["sha256"]:
        record["state"] = "sha256_mismatch"
        return record
    record["state"] = "verified"
    return record


def inspect_model_asset_contract(
    project_root: str | Path,
    primary_model_path: str | Path | None,
) -> dict[str, Any]:
    """Describe model readiness without creating any runtime/hardware object.

    The primary detector is required when configured.  The fallback detector
    remains explicitly optional until a separately authorized, versioned
    artifact is added to the lock and archive.
    """
    root = Path(project_root).resolve()
    lock_path = root / LOCK_RELATIVE_PATH
    payload, lock = _load_lock(lock_path)
    if primary_model_path is None:
        return {
            "schema_version": 1,
            "contract_id": payload.get("contract_id") if payload else None,
            "lock": lock,
            "assets": [],
            "optional_unavailable": [],
            "state": "not_configured",
        }

    entries = _entries_by_role(payload)
    primary = _asset_record(
        role="primary_detector",
        required=True,
        project_root=root,
        configured_path=_resolved_path(root, primary_model_path),
        lock_path=lock_path,
        entry=entries.get("primary_detector"),
        lock_available=payload is not None,
    )
    fallback = _asset_record(
        role="fallback_detector",
        required=False,
        project_root=root,
        configured_path=_resolved_path(root, FALLBACK_RELATIVE_PATH),
        lock_path=lock_path,
        entry=entries.get("fallback_detector"),
        lock_available=payload is not None,
    )
    assets = [primary, fallback]
    required_invalid = [
        asset for asset in assets if asset["required"] and asset["state"] != "verified"
    ]
    optional_unavailable = [
        asset["role"]
        for asset in assets
        if not asset["required"] and asset["state"] != "verified"
    ]
    return {
        "schema_version": 1,
        "contract_id": payload.get("contract_id") if payload else None,
        "lock": lock,
        "assets": assets,
        "optional_unavailable": optional_unavailable,
        "state": "blocked" if required_invalid else "ready",
    }


def require_model_asset_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reject any unverified *required* asset before runtime initialization."""
    blocked = [
        asset
        for asset in contract.get("assets", [])
        if asset.get("required") and asset.get("state") != "verified"
    ]
    if not blocked:
        return contract
    states = "; ".join(
        f"{asset.get('role')}={asset.get('state')} ({asset.get('configured_path')})"
        for asset in blocked
    )
    lock_state = contract.get("lock", {}).get("state")
    raise ModelAssetContractError(
        "模型资产契约未就绪，拒绝在 Demo/SDK/相机/MoveIt 初始化前继续: "
        f"{states}；lock={lock_state}。请使用权威发布制品补齐或恢复 "
        f"{contract.get('lock', {}).get('path', LOCK_RELATIVE_PATH)} 中的 "
        "version、size_bytes、sha256、source_uri，并重新生成 exact-SHA archive；"
        "不得从 dirty checkout 借用模型。"
    )


def verified_asset_path(
    contract: Mapping[str, Any], role: str
) -> Path | None:
    """Return a path only for an integrity-verified locked asset."""
    for asset in contract.get("assets", []):
        if asset.get("role") == role and asset.get("state") == "verified":
            return Path(str(asset["path"]))
    return None
