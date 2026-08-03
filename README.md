# dual-arm-shelf-dispenser

Shelf picking and cross-layer placing on a dual Realman RM75 platform: a
head-mounted RGB-D camera finds a bottle in a shelf bin, MoveIt Task Constructor
plans the reach, the lift lowers the held bottle to another layer, and the arm
places it in an observed empty slot.

Carved out of the [Grabber](https://github.com/CSQ-AN94/Grabber) monorepo, which
served this, a table-top grasp demo and a side-table delivery flow from one
4950-line `demo.py` and one shared safety profile. Editing one broke another —
the post-pick carry pose regression came straight from a validator forcing the
shelf and table profiles to share a `home_joints_deg` taught three weeks earlier
for a different task.

**Status: real hardware, partially working.** The first successful real shelf
grasp landed 2026-08-03. A full pick → lift → place cycle has not completed. The
honest per-stage picture is in `docs/`.

## Hardware

| | |
|---|---|
| Arms | 2 × Realman RM75 (7-DoF), `robotic-arm` pip SDK |
| Gripper | RMG24, right arm |
| Depth | RealSense D435 (head) + wrist cameras |
| Lift | serial column, 250–707 mm |
| Compute | Jetson AGX Orin, ROS 2 Humble |

## Layout

```
shelf_dispenser/       the library
    orchestrator.py    one run's hardware, scene and planning state
    arm.py             one RealMan arm: connection, IK, motion primitives
    arm_worker.py      the second arm, in its own process
    safety.py          the electronic fence
    left_arm.py        the fence, expressed in the left arm's base frame
    ros/               entry points run by the system Python, by path
mtc_ws/                ROS 2 workspace, MoveIt Task Constructor planner (C++)
scripts/               one entry point per pipeline stage
test/                  the suite; run `pytest -q` from the repo root,
                       which includes mtc_ws's source-contract tests
```

Names describe what is behind the interface, not where the code came from.
`ros/` is the realest seam in the repo — those modules never share an
interpreter with their caller — and is named `ros` rather than `moveit` so it
cannot shadow MoveIt's own package.

`orchestrator.py` is 4950 lines and trips the god-module check in
`scripts/architecture_report.py`. That is the next real piece of work: the
perception cluster inside it is the most self-contained and would come out
first.

## The pipeline

Each stage re-samples its own inputs immediately before consuming them, and
writes an execution record the next stage checks.

```
normalize_to_grasp_start.py    both arms + lift to the taught start pose
calibrate_mtc_gripper.py       empty-close baseline, so "holding" is measurable
capture_mtc_direct_pick_scene.py   head RGB-D → voxels + YOLO bottle
plan_shelf_transfer (MTC)      approach, grasp, retreat
execute_mtc_trajectory.py pick
normalize_to_grasp_start.py --target carry_home   back to the start pose, holding
execute_mtc_lift_transfer.py   647 → 250 mm with the bottle held
capture_empty_shelf_places.py  find an empty slot on the lower layer
plan_shelf_transfer (MTC)
execute_mtc_trajectory.py place
```

## Safety

Nothing moves on a claim; every gate is checked against a live reading.

- **Electronic fence** — every dense trajectory point's TCP must be inside the
  workspace box and at least one allowed zone, and outside every keepout box.
- **Collision recheck** — the plan is re-validated against the live MoveIt scene
  after planning, not only during it.
- **Joint-limit margin** — measured against the arm's own starting excess, so a
  pose already outside the margin cannot deadlock every planner.
- **Gripper feedback** — "holding" means a close position above the measured
  empty-close baseline, not a commanded state.
- **Left-arm drift** — the passive arm is captured live into every plan's
  collision scene and required not to move during execution.

## Both arms

The RealMan SDK's `Algo` is process-global: install angle, tool frame and joint
limits are set on the library, not on a handle, so a second `RobotSession` in
one process silently overwrites the first one's kinematics. That is why the left
arm was read-only for months.

`shelf_dispenser/arm_worker.py` gives each arm its own process. A worker owns one
`RobotSession` and answers newline-delimited JSON on stdin; the parent holds an
`ArmProxy` exposing a whitelist of methods. The whitelist is the safety
boundary, not a convenience — a typo cannot reach a method nobody vetted for the
second arm.

`ros/plan_once.py` derives joint and link names from the planning group, so
the same collision-aware planning path serves `left_arm` and `right_arm`. The
two base frames are related by a measured transform (`config.yaml`,
`T_base_right_to_base_left`, 2026-07-14), which lets both arms be fenced in one
coordinate system rather than two.

## Running the tests

```bash
python -m pytest test/ -q
```

No robot, no ROS, no cameras required.

## Provenance

This is a portfolio and research repository, not a product. Where something is
unverified it says so, including in the safety profiles' own `provenance`
fields. Related: [realman-dual-arm-robot](https://github.com/CSQ-AN94/realman-dual-arm-robot)
(SDK teaching material), [vision-guided-bottle-grasp](https://github.com/CSQ-AN94/vision-guided-bottle-grasp)
(the single-bottle table demo this grew out of).
