from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "shelf_row_template_tool.py"
SPEC = importlib.util.spec_from_file_location("shelf_row_template_tool", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _pose(position: float):
    return {
        "xyz_moveit": [position, 0.5, -0.1],
        "quat_xyzw_moveit": [0.0, 0.0, 0.0, 1.0],
        "reference_joints_deg": [position * 100.0] * 7,
    }


def _complete_map():
    payload = MODULE.create_anchor_map(
        row_start_m=0.0,
        row_end_m=0.10,
        spacing_m=0.01,
        overlap_start_m=0.04,
        overlap_end_m=0.06,
        layers=(("upper", 647), ("lower", 250)),
    )
    for layer in ("upper", "lower"):
        for arm, positions in (
            ("left_arm", (0.0, 0.06)),
            ("right_arm", (0.04, 0.10)),
        ):
            for position in positions:
                MODULE.add_anchor(
                    payload,
                    layer_id=layer,
                    position_m=position,
                    arm_id=arm,
                    operations=("pick", "place"),
                    source="test",
                    **_pose(position),
                )
    return payload


def test_two_layers_generate_one_centimetre_pick_and_place_map():
    generated = MODULE.generate_templates(_complete_map())

    assert generated["schema_version"] == MODULE.TEMPLATE_SCHEMA
    assert generated["point_count"] == 22
    assert generated["execution_verified"] is False
    assert generated["gripper_contract"]["included_in_tcp_pose"] is False

    upper = [point for point in generated["points"] if point["layer_id"] == "upper"]
    assert len(upper) == 11
    assert upper[3]["eligible_arms"] == ["left_arm"]
    assert upper[4]["eligible_arms"] == ["left_arm", "right_arm"]
    assert upper[6]["eligible_arms"] == ["left_arm", "right_arm"]
    assert upper[7]["eligible_arms"] == ["right_arm"]
    assert set(upper[2]["operations"]) == {"pick", "place"}
    pick = upper[2]["operations"]["pick"]["arm_candidates"]["left_arm"]
    assert pick["tcp_pose_moveit"]["xyz"][0] == pytest.approx(0.02)
    assert pick["requires_gripper_clearance_validation"] is True
    assert "gripper" not in pick["tcp_pose_moveit"]

    overlap_pick = upper[5]["operations"]["pick"]
    assert overlap_pick["arm_selection"] == "runtime_plan_both_choose_valid_best"
    assert set(overlap_pick["arm_candidates"]) == {"left_arm", "right_arm"}
    assert overlap_pick["pose_semantics"] == "demonstrated_grasp_curve"
    place = upper[5]["operations"]["place"]
    assert place["pose_semantics"] == "grasp_curve_place_initial_guess_only"
    assert place["release_allowed_without_runtime_adjustment"] is False
    assert place["required_runtime_adjustment"] == (
        "fresh_support_surface_height_and_collision_checked_place_plan"
    )


def test_generated_reference_joints_come_from_nearest_anchor_not_interpolation():
    generated = MODULE.generate_templates(_complete_map())
    point = next(
        item
        for item in generated["points"]
        if item["layer_id"] == "upper" and item["row_position_m"] == 0.03
    )
    candidate = point["operations"]["pick"]["arm_candidates"]["left_arm"]
    assert candidate["reference_joints_deg"] == pytest.approx([0.0] * 7)


def test_generation_learns_spatial_order_when_nominal_labels_are_noisy():
    payload = _complete_map()
    for operation in MODULE.OPERATIONS:
        payload["layers"]["upper"]["anchors"]["right_arm"][operation] = []
    for label, actual_x in (
        (0.04, 0.10),
        (0.06, 0.04),  # Nominal order is noisy: this was placed after 0.08.
        (0.08, 0.07),
        (0.10, 0.01),
    ):
        MODULE.add_anchor(
            payload,
            layer_id="upper",
            position_m=label,
            arm_id="right_arm",
            operations=MODULE.OPERATIONS,
            xyz_moveit=[actual_x, 0.5, -0.1],
            quat_xyzw_moveit=[0.0, 0.0, 0.0, 1.0],
            reference_joints_deg=[actual_x * 100.0] * 7,
            source="noisy_ruler_test",
        )

    generated = MODULE.generate_templates(payload)
    by_position = {
        point["row_position_m"]: point
        for point in generated["points"]
        if point["layer_id"] == "upper"
    }
    learned_x = [
        by_position[position]["operations"]["pick"]["arm_candidates"][
            "right_arm"
        ]["tcp_pose_moveit"]["xyz"][0]
        for position in (0.04, 0.06, 0.08, 0.10)
    ]

    assert learned_x == pytest.approx([0.10, 0.07, 0.04, 0.01])


def test_arm_assignment_is_enforced_when_recording_anchor():
    payload = _complete_map()
    with pytest.raises(MODULE.SafetyAbort, match="只允许 left_arm"):
        MODULE.add_anchor(
            payload,
            layer_id="upper",
            position_m=0.02,
            arm_id="right_arm",
            operations=("pick",),
            source="test",
            **_pose(0.02),
        )


def test_generation_refuses_missing_operation_coverage():
    payload = _complete_map()
    payload["layers"]["lower"]["anchors"]["left_arm"]["place"] = []
    with pytest.raises(MODULE.SafetyAbort, match="至少需要两个锚点"):
        MODULE.generate_templates(payload)


def test_generation_uses_observed_spatial_span_when_nominal_edge_was_skipped():
    payload = _complete_map()
    payload["layers"]["upper"]["anchors"]["right_arm"]["pick"][0][
        "position_m"
    ] = 0.05

    generated = MODULE.generate_templates(payload)
    edge = next(
        point
        for point in generated["points"]
        if point["layer_id"] == "upper" and point["row_position_m"] == 0.04
    )

    candidate = edge["operations"]["pick"]["arm_candidates"]["right_arm"]
    assert candidate["tcp_pose_moveit"]["xyz"][0] == pytest.approx(0.04)


def test_guided_teach_records_each_pose_and_generates_map(
    tmp_path, monkeypatch
):
    anchors = tmp_path / "row_anchors.json"
    templates = tmp_path / "row_templates.json"
    captured = []

    def fake_capture(cli, payload, arm_id, expected_lift_mm):
        captured.append((arm_id, expected_lift_mm))
        index = len(captured)
        return (
            [index / 100.0, 0.5, -0.1],
            [0.0, 0.0, 0.0, 1.0],
            [float(index)] * 7,
        )

    monkeypatch.setattr(MODULE, "_capture_live", fake_capture)
    monkeypatch.setattr("builtins.input", lambda _: "")
    cli = SimpleNamespace(
        anchors=anchors,
        templates_output=templates,
        row_start_m=0.0,
        row_end_m=0.10,
        spacing_m=0.01,
        overlap_start_m=0.04,
        overlap_end_m=0.06,
        layer=[("upper", 647), ("lower", 250)],
        anchors_per_arm=2,
        samples=2,
        sample_gap_s=0.0,
        max_drift_deg=0.5,
    )

    assert MODULE._teach(cli) == 0
    assert len(captured) == 8
    assert [arm for arm, _ in captured] == [
        "right_arm",
        "right_arm",
        "right_arm",
        "right_arm",
        "left_arm",
        "left_arm",
        "left_arm",
        "left_arm",
    ]
    generated = MODULE.json.loads(templates.read_text(encoding="utf-8"))
    assert generated["point_count"] == 22
    overlap = generated["points"][5]
    assert overlap["eligible_arms"] == ["left_arm", "right_arm"]


def test_teach_defaults_to_five_centimetre_anchor_density():
    cli = MODULE._build_parser().parse_args(
        ["teach", "outputs/row_anchors.json"]
    )

    # The default right/left teaching regions are both 0.35 m wide, so
    # endpoints plus every 0.05 m require eight anchors per arm and layer.
    assert cli.anchors_per_arm == 8


def test_guided_teach_can_start_with_selected_layer(
    tmp_path, monkeypatch
):
    captured = []

    def fake_capture(cli, payload, arm_id, expected_lift_mm):
        captured.append((arm_id, expected_lift_mm))
        return [0.0, 0.5, -0.1], [0.0, 0.0, 0.0, 1.0], [0.0] * 7

    responses = iter(("2", "", "q"))
    monkeypatch.setattr(MODULE, "_capture_live", fake_capture)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    cli = SimpleNamespace(
        anchors=tmp_path / "row_anchors.json",
        templates_output=None,
        row_start_m=0.0,
        row_end_m=0.10,
        spacing_m=0.01,
        overlap_start_m=0.04,
        overlap_end_m=0.06,
        layer=[("upper", 647), ("lower", 250)],
        anchors_per_arm=2,
        samples=2,
        sample_gap_s=0.0,
        max_drift_deg=0.5,
    )

    assert MODULE._teach(cli) == 0
    assert captured == [("right_arm", 250)]


def test_guided_teach_reports_failure_and_retries_same_point(
    tmp_path, monkeypatch, capsys
):
    anchors = tmp_path / "row_anchors.json"
    calls = 0

    def flaky_capture(cli, payload, arm_id, expected_lift_mm):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MODULE.SafetyAbort("机械臂仍在运动")
        return [0.0, 0.5, -0.1], [0.0, 0.0, 0.0, 1.0], [0.0] * 7

    responses = iter(("", "", "q"))
    monkeypatch.setattr(MODULE, "_capture_live", flaky_capture)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    cli = SimpleNamespace(
        anchors=anchors,
        templates_output=None,
        row_start_m=0.0,
        row_end_m=0.10,
        spacing_m=0.01,
        overlap_start_m=0.04,
        overlap_end_m=0.06,
        layer=[("only", 647)],
        anchors_per_arm=2,
        samples=2,
        sample_gap_s=0.0,
        max_drift_deg=0.5,
    )

    assert MODULE._teach(cli) == 0
    output = capsys.readouterr().out
    assert "记录失败：机械臂仍在运动" in output
    assert "记录成功并已保存" in output
    assert calls == 2


def test_guided_teach_opens_requested_gripper_then_records(
    tmp_path, monkeypatch, capsys
):
    anchors = tmp_path / "row_anchors.json"
    opened = []
    captured = []

    def fake_open(cli, arm_id):
        opened.append(arm_id)
        return {"pos": [900]}

    def fake_capture(cli, payload, arm_id, expected_lift_mm):
        captured.append(arm_id)
        return [0.0, 0.5, -0.1], [0.0, 0.0, 0.0, 1.0], [0.0] * 7

    responses = iter(("o", "", "q"))
    monkeypatch.setattr(MODULE, "_command_live_gripper", fake_open)
    monkeypatch.setattr(MODULE, "_capture_live", fake_capture)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    cli = SimpleNamespace(
        anchors=anchors,
        templates_output=None,
        row_start_m=0.0,
        row_end_m=0.10,
        spacing_m=0.01,
        overlap_start_m=0.04,
        overlap_end_m=0.06,
        layer=[("only", 647)],
        anchors_per_arm=2,
        samples=2,
        sample_gap_s=0.0,
        max_drift_deg=0.5,
    )

    assert MODULE._teach(cli) == 0
    output = capsys.readouterr().out
    assert opened == ["right_arm"]
    assert captured == ["right_arm"]
    assert "夹爪已打开并通过反馈检查" in output


def test_left_gripper_open_does_not_serialize_demo_params(monkeypatch):
    opened_with = []

    class FakeRobot:
        def assert_arm_healthy(self):
            pass

        def joints_deg(self):
            return [0.0] * 7

        def tcp_from_joints(self, joints):
            return MODULE.np.eye(4)

        def open_gripper(self, *args):
            opened_with.append(args)
            return {"pos": [900]}

        def close(self):
            pass

    class FakeView:
        def assert_tcp_point(self, point, *, label):
            pass

    monkeypatch.setattr(MODULE, "load_config", lambda _: object())
    monkeypatch.setattr(MODULE, "load_safety_profile", lambda *a, **k: object())
    monkeypatch.setattr(
        MODULE, "open_live_arm", lambda *a, **k: (FakeRobot(), FakeView())
    )

    state = MODULE._command_live_gripper(
        SimpleNamespace(config="config.yaml", safety_config="safety.json"),
        "left_arm",
    )

    assert state == {"pos": [900]}
    assert opened_with == [()]


def test_gripper_subcommands_default_to_the_left_arm_and_carry_the_verb(
    monkeypatch, capsys
):
    asked = []
    monkeypatch.setattr(
        MODULE,
        "_command_live_gripper",
        lambda cli, arm, *, close=False: (
            asked.append((arm, close)) or {"pos": [0 if close else 900]}
        ),
    )

    assert MODULE.main(["open-gripper"]) == 0
    assert MODULE.main(["close-gripper", "--arm", "right_arm"]) == 0

    assert asked == [("left_arm", False), ("right_arm", True)]
    out = capsys.readouterr().out
    assert "左" not in out  # arms are named by id, not by side
    assert "夹爪已打开：pos=900" in out
    assert "夹爪已收拢：pos=0" in out


def test_left_arm_worker_allows_the_whole_gripper_cycle():
    from shelf_dispenser.arm_worker import ALLOWED_METHODS

    # close_gripper without calibrate_empty_close is unusable: the grasp
    # judgment refuses outright without this round's measured baseline.
    for method in (
        "open_gripper",
        "close_gripper",
        "close_empty_gripper",
        "calibrate_empty_close",
        "gripper_state",
    ):
        assert method in ALLOWED_METHODS
