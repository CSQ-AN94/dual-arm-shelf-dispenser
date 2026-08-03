"""Offline exact-archive checks for the bottle-detector model contract.

These tests deliberately use a temporary Git repository and ``git archive``:
the release path must reject a missing or altered model before any Demo,
camera, SDK, MoveIt, or robot object is constructed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.model_assets import (
    ModelAssetContractError,
    inspect_model_asset_contract,
    require_model_asset_contract,
)
from shelf_dispenser.run_manifest import build_run_manifest


PRIMARY = "intelligence/yolo_models/8_17.pt"
FALLBACK = "intelligence/yolo_models/yolo11n.pt"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _asset_entry(
    role: str, relative_path: str, payload: bytes, *, required: bool
) -> dict:
    return {
        "role": role,
        "relative_path": relative_path,
        "required": required,
        "version": f"test-{role}-v1",
        "size_bytes": len(payload),
        "sha256": _sha256(payload),
        "source_uri": f"artifact://tests/{role}-v1",
    }


def _archive_release(
    tmp_path: Path,
    *,
    include_primary: bool = True,
    include_fallback: bool = False,
) -> Path:
    """Create and extract a tiny exact-Git release without any transport."""
    source = tmp_path / "source"
    source.mkdir()
    primary_payload = b"primary-model-v1"
    fallback_payload = b"fallback-model-v1"
    lock = {
        "schema_version": 1,
        "contract_id": "bottle-grasp-yolo-models",
        "assets": [
            _asset_entry(
                "primary_detector", PRIMARY, primary_payload, required=True
            ),
            {
                "role": "fallback_detector",
                "relative_path": FALLBACK,
                "required": False,
                "version": None,
                "size_bytes": None,
                "sha256": None,
                "source_uri": None,
            },
        ],
    }
    (source / "shelf_dispenser").mkdir()
    (source / "shelf_dispenser" / "model_assets.lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    (source / "config.yaml").write_text(
        f"vision:\n  model_path: {PRIMARY}\n", encoding="utf-8"
    )

    for relative_path, payload, include in (
        (PRIMARY, primary_payload, include_primary),
        (FALLBACK, fallback_payload, include_fallback),
    ):
        if include:
            destination = source / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)

    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=asset-contract-test",
            "-c",
            "user.email=asset-contract-test@example.invalid",
            "commit",
            "-qm",
            "archive fixture",
        ],
        cwd=source,
        check=True,
    )
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=source,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    archive_path = tmp_path / "release.tar"
    archive_path.write_bytes(archive)
    with tarfile.open(archive_path) as bundle:
        bundle.extractall(deployment)
    return deployment


def test_exact_archive_with_verified_models_passes_pure_preflight(tmp_path):
    deployment = _archive_release(tmp_path)

    contract = inspect_model_asset_contract(deployment, PRIMARY)

    assert contract["state"] == "ready"
    assert contract["assets"][0]["state"] == "verified"
    assert contract["assets"][1]["state"] == "not_registered"
    assert contract["optional_unavailable"] == ["fallback_detector"]
    assert require_model_asset_contract(contract) is contract


def test_exact_archive_rejects_missing_model_before_runtime_initialization(tmp_path):
    deployment = _archive_release(tmp_path, include_primary=False)

    contract = inspect_model_asset_contract(deployment, PRIMARY)

    assert contract["state"] == "blocked"
    assert contract["assets"][0]["state"] == "missing"
    with pytest.raises(ModelAssetContractError, match="primary_detector.*missing"):
        require_model_asset_contract(contract)


def test_exact_archive_rejects_hash_mismatch_before_runtime_initialization(tmp_path):
    deployment = _archive_release(tmp_path)
    # Keep the byte count unchanged so this exercises the SHA gate itself.
    (deployment / PRIMARY).write_bytes(b"primary-model-x1")

    contract = inspect_model_asset_contract(deployment, PRIMARY)

    assert contract["state"] == "blocked"
    assert contract["assets"][0]["state"] == "sha256_mismatch"
    with pytest.raises(
        ModelAssetContractError, match="primary_detector.*sha256_mismatch"
    ):
        require_model_asset_contract(contract)


@pytest.mark.parametrize(
    ("mutation", "lock_state"),
    [
        (
            lambda payload: payload.update(contract_id="wrong-contract"),
            "invalid_contract_id",
        ),
        (
            lambda payload: payload["assets"].append(dict(payload["assets"][0])),
            "duplicate_asset_roles",
        ),
    ],
)
def test_asset_lock_rejects_wrong_contract_or_duplicate_roles(
    tmp_path, mutation, lock_state
):
    deployment = _archive_release(tmp_path)
    lock_path = deployment / "shelf_dispenser" / "model_assets.lock.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    mutation(payload)
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    contract = inspect_model_asset_contract(deployment, PRIMARY)

    assert contract["state"] == "blocked"
    assert contract["lock"]["state"] == lock_state
    with pytest.raises(ModelAssetContractError, match="primary_detector"):
        require_model_asset_contract(contract)


def test_manifest_records_primary_verification_and_optional_fallback_state(tmp_path):
    deployment = _archive_release(tmp_path)
    args = SimpleNamespace(config=str(deployment / "config.yaml"))
    config = SimpleNamespace(
        vision=SimpleNamespace(model_path=PRIMARY), calibration=None
    )

    manifest = build_run_manifest(
        args=args,
        config=config,
        project_root=deployment,
        environ={},
    )

    assets = manifest["artifacts"]["model_assets"]
    assert assets["state"] == "ready"
    assert assets["assets"][0]["state"] == "verified"
    assert assets["optional_unavailable"] == ["fallback_detector"]


@pytest.mark.parametrize(
    ("resume_at_wrist", "finish_from_current"),
    [(False, False), (True, False), (False, True)],
)
def test_demo_rejects_missing_primary_before_safety_or_hardware_setup(
    tmp_path, monkeypatch, resume_at_wrist, finish_from_current
):
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.project_root = tmp_path
    demo.cfg = SimpleNamespace(vision=SimpleNamespace(model_path=PRIMARY))
    demo.args = SimpleNamespace(
        task_mode=None,
        resume_at_wrist=resume_at_wrist,
        finish_from_current=finish_from_current,
        execute=False,
        stop_after_observation=False,
    )
    monkeypatch.setattr(
        "shelf_dispenser.orchestrator.load_safety_profile",
        lambda *_args, **_kwargs: pytest.fail(
            "safety/hardware-adjacent initialization must not run"
        ),
    )

    with pytest.raises(SafetyAbort, match="Demo/SDK/相机/MoveIt"):
        demo.initialize()


def test_initialize_passes_verified_absolute_primary_to_detector_after_chdir(
    tmp_path, monkeypatch
):
    deployment = _archive_release(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    detector_calls = []
    camera_calls = []

    class FakeSafety:
        name = "test-profile"
        grasp_height_fraction = None

        @staticmethod
        def moveit_collision_boxes():
            return []

    class FakeDashboard:
        def __init__(self, *_args):
            pass

        def start(self):
            pass

    def fake_detector(model_path, confidence, **kwargs):
        detector_calls.append((model_path, confidence, kwargs))
        return object()

    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.project_root = deployment
    demo.model_asset_contract = None
    demo.cfg = SimpleNamespace(vision=SimpleNamespace(model_path=PRIMARY))
    demo.args = SimpleNamespace(
        task_mode=None,
        resume_at_wrist=False,
        finish_from_current=False,
        execute=False,
        stop_after_observation=False,
        plan_only=False,
        safety_config="unused",
        safety_profile="test-profile",
        dispense=False,
        host="127.0.0.1",
        port=8879,
    )
    demo.params = DemoParams()
    demo.state = object()
    demo._ensure_head_reference = lambda: None
    demo._start_camera = lambda name: camera_calls.append(name)
    demo.stage = lambda *_args, **_kwargs: None
    monkeypatch.setattr(
        "shelf_dispenser.orchestrator.load_safety_profile",
        lambda *_args, **_kwargs: FakeSafety(),
    )
    monkeypatch.setattr("shelf_dispenser.orchestrator.Dashboard", FakeDashboard)
    monkeypatch.setattr("shelf_dispenser.orchestrator.BottleDetector", fake_detector)
    monkeypatch.chdir(outside)

    demo.initialize()

    assert detector_calls == [
        (
            str((deployment / PRIMARY).resolve()),
            demo.params.confidence,
            {"fallback_model_path": None, "fallback_confidence": 0.05},
        )
    ]
    assert camera_calls == ["head"]


def test_wrist_uses_verified_primary_when_optional_fallback_is_unavailable(
    tmp_path, monkeypatch
):
    deployment = _archive_release(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    calls = []
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.model_asset_contract = inspect_model_asset_contract(deployment, PRIMARY)
    monkeypatch.setattr(
        "shelf_dispenser.orchestrator.BottleDetector",
        lambda model_path, confidence: calls.append((model_path, confidence))
        or object(),
    )
    monkeypatch.chdir(outside)

    detector = demo._wrist_detector_from_contract()

    assert detector is not None
    assert calls == [(str((deployment / PRIMARY).resolve()), 0.05)]
