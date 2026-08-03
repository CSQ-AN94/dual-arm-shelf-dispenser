# grabber_mtc_planner

RM75 bottle planning with MoveIt Task Constructor. The historical dual-arm
full transfer remains the default; `mode: pick_only` adds a fixed-head direct
pick that stops after source retreat and defaults to the right arm. The same
pipeline can select the left arm for plan-only feasibility checks.
`mode: place_only` plans a held right-arm bottle to a fresh empty-shelf
candidate and stops after release/retreat.

**Plan only.** The node has no execution code path: no `Task::execute()`, no
`FollowJointTrajectory`, no RealMan SDK, no lift, no chassis. `--plan-only` is a
mandatory flag so a reader of the command line can see that.

## Task structure

```
current_state                      (live PlanningScene + live joint states)
spawn_scenario_objects             (bottle + shelf boxes, optional)
Alternatives
├── full transfer: right_arm_branch / left_arm_branch
├── pick-only: <planning-arm>__<grasp-candidate> branches
└── place-only: right_arm__place branch
```

Each branch, in order:

| stage | solver | meaning |
|---|---|---|
| `allow_support_contact` | — | allow only bottle↔measured support contact |
| `connect_to_source_pregrasp` | OMPL (RRTConnect) | current state → start of the approach |
| `source_pick/source_pregrasp_ik` | ComputeIK (multi-solution) | collision-free pregrasp for one final TCP candidate |
| `source_pick/source_approach` | Cartesian | collision-checked approach, stopping before hand contact |
| `source_pick/allow_final_grasp_contact` | — | allow bottle contact only for this arm's touch links |
| `source_pick/source_contact` | Cartesian | final short contact segment |
| `source_pick/attach_bottle` | — | bottle becomes an attached object |
| `source_pick/source_lift` | Cartesian | pick-only: lift vertically off the support |
| `source_pick/forbid_support_contact_after_lift` | — | pick-only: restore bottle↔support collision checks |
| `source_pick/source_retreat` | Cartesian | straight line back out of the shelf |
| `source_pick/restore_bottle_collision_check` | — | pick-only: restore the temporary bottle↔touch-link ACM entries |
| `transport` | OMPL (RRTConnect) | carry the bottle to the target preplace |
| `target_place/target_insert` | Cartesian | straight line into the target shelf |
| `target_place/target_place_ik` | ComputeIK (multi-solution) | place pose |
| `target_place/open_gripper_semantic` | — | scene-only release marker |
| `target_place/detach_bottle` | — | bottle returns to the world |
| `target_place/target_retreat` | Cartesian | gripper leaves the target shelf |
| `target_place/restore_bottle_collision_check` | — | bottle↔touch-link collisions are forbidden again |

A branch is `solved` only when the **whole** container produced a solution. A
successful source grasp with a failed transport or place reports
`solved: false`. `earliest_failure_stage` is the first leaf that recorded an
actual failed attempt; empty stages and connectors that never ran are omitted.

The bottle/hand ACM relaxation starts only after the collision-checked
current→pregrasp and long approach segments. It is limited to that arm's
configured touch links and restored at the branch end. Dynamic non-target
voxels are never entered in the ACM. The attached bottle therefore continues
to collide with shelf geometry, the robot body, the other arm, and every
non-target bottle.

MTC's `attachObject()` does not copy touch links into the attached body, so
the attach stage explicitly copies the same narrow configured set. Pick-only
can then restore its temporary ACM entry after retreat while the attached-body
touch-link semantics preserve only the intended held-bottle contact. Full
transfer restores its ACM entry after detach.

`source_contact_distance_m` and `source_lift_distance_m` are commissioning
knobs for the final allowed-contact segment and vertical support clearance.
The full-transfer fixture keeps the checked-in 20 mm contact default; the
live pick-only converter uses 70 mm because the `r_hand` collision mesh
extends about 94 mm beyond the active TCP. Pick-only lifts 50 mm before the
horizontal exit. These values are not proof of execution safety.

## Arm selection

Full transfer plans both arms. Pick-only plans only `planning_arm_id`
(`right_arm` by default) and selects the lowest-cost complete grasp candidate.
Selection is hard feasibility first, then execution eligibility, then MTC
total cost. Arm identity remains in `config/dual_rm75_arms.yaml`.

`execution_eligible` never skips planning. The left arm is currently
`execution_eligible: false` /
`execution_block_reason: LEFT_TOOL_CALIBRATION_REQUIRED`; its feasibility is
still reported, but a complete calibrated right-arm branch is preferred.

## Build

```bash
cd ~/mtc_ws && colcon build --packages-select grabber_mtc_planner --symlink-install
```

Requires `ros-humble-moveit-task-constructor-core` (or a source build of MTC in
the same workspace) and the installed `dual_rm_75b_moveit_config`.

## Run (plan only)

With `move_group` already running, the experimental launch is:

```bash
ros2 launch grabber_mtc_planner plan_shelf_transfer_experimental.launch.py scenario:=<path>.yaml out:=mtc_plan_result.json hold_seconds:=300
```

`hold_seconds` keeps the node alive so RViz's *Motion Planning Tasks* panel can
browse the per-stage solutions; use `0` for a batch run.

`ros2 run grabber_mtc_planner plan_shelf_transfer --plan-only --scenario ...`
works too, but only if you pass `robot_description` and friends yourself. The
50 ms KDL timeout override exists only in the experimental launch; it does not
change the installed MoveIt configuration or direct node runs.

Exit codes: `0` solved, `1` no complete solution, `2` bad input/setup,
`3` task init failed.

## Fixed-head direct pick (plan-only)

The new entry has no observation pose and no wrist-camera dependency. It
captures only the fixed head camera and does not connect either arm:

```bash
python3 scripts/capture_mtc_direct_pick_scene.py \
  --planning-arm left_arm \
  --scenario-out /tmp/mtc_direct_pick.yaml
```

Omit `--planning-arm` for the unchanged right-arm default. The left-arm path
does not reuse the right-arm taught staging joints and remains plan-only until
its tool/TCP calibration is measured.

The emitted scenario contains the fresh base-frame target, two horizontal-jaw
horizontal-finger TCP candidates (`horizontal_fingers_roll_0`,
`horizontal_fingers_roll_180`), the measured
`fence_shelf_{bottom,top,back}` boxes, and every head RGB-D voxel except the
locked target occupancy. Plan it with the unchanged public experimental
launcher:

```bash
ros2 launch grabber_mtc_planner plan_shelf_transfer_experimental.launch.py \
  scenario:=/tmp/mtc_direct_pick.yaml \
  out:=/tmp/mtc_pick_result.json \
  hold_seconds:=0
```

A solved pick-only run writes
`/tmp/mtc_pick_result.json.trajectory.json`. The export contains exactly seven
ordered joints for the selected arm, positions and velocities in degrees, plus
`pregrasp/approach/attach/retreat` boundaries and gripper events. The retreat
phase contains the vertical lift, collision-checked shelf exit, and an OMPL
move to the taught `shelf_template.home_joints_deg` carry pose while the
bottle remains attached.
Validate the contract offline:

```bash
python3 scripts/validate_mtc_pick_trajectory.py \
  /tmp/mtc_pick_result.json.trajectory.json
```

The MTC node deliberately remains plan-only with `execution_supported: false`.
Right-arm exports use `EXTERNAL_GATED_PICK_EXECUTOR_REQUIRED`; left-arm exports
use `LEFT_TOOL_CALIBRATION_REQUIRED`, and the real execution bridge rejects
them. The validator checks the selected-arm joint order, phase boundaries,
current-state/trajectory-start match,
scene and target freshness, and the required open/close feedback evidence. It
does not send robot commands.

The separate explicit executor binds result + scenario + trajectory, verifies
the live lift and both-arm start state, hardware health, controller fence,
freshness, electronic-fence TCP path, and gripper feedback, then closes/opens
only at the exported boundary:

```bash
python3 scripts/execute_mtc_trajectory.py pick \
  --result /tmp/mtc_pick_result.json \
  --trajectory /tmp/mtc_pick_result.json.trajectory.json \
  --scenario /tmp/mtc_direct_pick.yaml

# Only after the dry validation above:
python3 scripts/calibrate_mtc_gripper.py \
  --record /tmp/gripper_calibration.json --execute
python3 scripts/execute_mtc_trajectory.py pick \
  --result /tmp/mtc_pick_result.json \
  --trajectory /tmp/mtc_pick_result.json.trajectory.json \
  --scenario /tmp/mtc_direct_pick.yaml \
  --gripper-calibration-record /tmp/gripper_calibration.json \
  --execute --allow-sdk-retiming --speed 100
```

RealMan receives one connected, blended sequence on each side of the gripper
event. `--allow-sdk-retiming` is mandatory because this adapter does not
reproduce MTC `time_from_start`; this is not a FollowJointTrajectory/streaming
controller claim, and the execution record states
`trajectory_timing_preserved: false`.

The complete real cross-layer runner starts the read-only live-state/MoveIt
stack, performs fixed-head pick planning and execution, lowers the stationary
compact dual-arm platform from 647 mm to 250 mm, captures the lower shelf, selects an
empty patch, replans, and places:

```bash
python3 scripts/run_mtc_cross_layer_workflow.py \
  --execute --allow-sdk-retiming \
  --operator-confirms-lower-shelf-obstacles-complete
```

It intentionally refuses at the lift step while
`shelf_dispenser/lift_transfer_647_to_250.json` remains `verified: false`.
That contract needs one supervised physical 647→250 mm clearance run with the
right arm at the taught carry pose and a recorded safe left-arm pose. The
program will not infer that missing chassis-clearance measurement from MuJoCo.

Place-only exports use the same blocked contract and
`transport/approach/release/retreat` boundaries:

```bash
python3 scripts/validate_mtc_place_trajectory.py \
  /tmp/mtc_place_result.json.trajectory.json
```

## Optional historical MuJoCo tools (not a real-scene gate)

These tools are retained for offline demos, replay regressions, and historical
failure reproduction. They are not part of the real-shelf workflow, not a
precondition for real plan-only review or motion, and not included in the core
acceptance result. Missing optional simulation meshes must not block the
fixed-head scene → live CurrentState → PlanningScene → MTC → safety-audit path.

The full digital twin imports the installed dual-RM75 STL/URDF, adds both
RMG24 two-finger grippers, the fixed head camera, the measured five-level
shelf shape, and scene-configured bottles. MuJoCo ground-truth segmentation
through the physical head camera gates replay of a trajectory
through pregrasp, approach, attach, retreat, transport, place, release, and
target retreat. It is not a live perception-to-planning loop. The default
viewer is the physical head camera; `--observer-view` explicitly enables a
mouse-adjustable external debug camera while perception remains on the head
camera. Viewer sidebars start collapsed. It never imports the RealMan SDK or
opens a robot socket.

The replay uses the exact MoveIt-expanded URDF (not the stale static URDF),
requires the gripper approach/finger axes to be within 5 degrees of a square
shelf-facing grasp, limits horizontal centring, vertical grasp height and
bottle-surface error, and requires release within 10 mm of the scenario target.

The description package is not committed because its upstream package has no
declared redistribution license. Fetch it read-only from the installed robot:

```bash
scripts/fetch_mujoco_robot_assets.sh
```

Generate a deterministic random two-layer scene with three third-layer
bottles and one or two lower-layer bottles:

```bash
conda activate robo
GRABBER_SIM_SEED=7 python scripts/mujoco_random_shelf_workflow.py
```

This command now writes a scene and a blocked manifest, not a trajectory.
The previous code silently warped an old MTC trajectory with Cartesian IK and
omitted the board between shelf layers; that output is no longer accepted by
`mujoco_full_workflow.py`. A random cross-layer replay must first obtain real
MTC pick-only and place-only exports for this exact scene and insert the
stationary-arm 647→250 mm lift segment.

The generator prints and stores the provisional offline workspace region in
`placement_selection.offline_workspace_region`. Its absolute coordinates are
metres in `platform_base_link`; the shelf centre is `(x=0.03, y=-0.72)`.
For physical layout, face the shelf from the robot and measure bottle
**centres**: the current continuous test rectangle is 4--33 cm rightward from
the left inside edge and 5--9 cm inward from the robot-facing front edge.
There are no slots. Bottle centres are sampled continuously, with at least
10 cm lateral separation: 6.6 cm bottle diameter / about 7 cm open inner jaw
width plus 3 cm margin. This keeps the geometric grasp corridors separate,
but is not a head-camera occlusion proof. The selected lower-layer centre
remains at least 13 cm from every other bottle centre. The 5 cm near boundary
is the 3.3 cm bottle radius plus 1.7 cm edge margin. The manifest also records
the mandatory +397 mm Z rebase from the 647 mm visualization frame to the
250 mm place-planning frame. These values are simulation hypotheses, not
real-robot calibration; MTC batch success and shelf registration must pass
before hardware use.

The old checked MTC artifact under
`artifacts/mujoco_full_workflow/` records the obsolete vertical-gripper
candidate and is now expected to fail the candidate/alignment gates. A new
MTC result must replace it before MTC replay can be claimed. Current
attach/release remains a bilateral-contact checked kinematic replay, not a
force-closure, controller-timing, or real-robot claim.

`mujoco_grabber_sim.py` remains the small primitive-based pick-only/place-only
contract smoke test. On macOS use `mjpython` whenever rendering or opening the
interactive viewer.

## Scenario

`scenarios/shelf_transfer_fixture.yaml` is a **fixture** (`fixture_source: true`):
the poses are the 2026-07-20 on-site shelf measurement from
`shelf_dispenser/safety_profiles.json` pushed through that profile's verified
`T_moveit_from_profile` bridge into `platform_base_link`. It is not live
perception and must be re-measured before anyone asks for real motion.

`scenarios/right_arm_placeback_trace.yaml` is a second historical fixture from
the 2026-07-22 evidence bundle where the right arm physically grasped and
lifted the bottle. It reproduces the proven TCP orientation and table keepout,
and reached a 10/10 complete right-arm plan-only rate against live joint state
on 2026-07-27. It is not a current bottle localization.

For a fresh **plan-only** bottle pose, reuse the existing safety-profile
transform instead of hand-editing coordinates:

```bash
python3 scripts/localization_to_mtc_scenario.py \
  outputs/shelf_dispenser/<run>/右腕精定位_localization.json \
  /tmp/localized_mtc.yaml \
  --template mtc_ws/src/grabber_mtc_planner/scenarios/right_arm_placeback_trace.yaml \
  --safety-profiles shelf_dispenser/safety_profiles.json \
  --profile table_demo
```

The converter checks the same localization quality limits as the grasp demo,
rejects inputs older than five minutes, and records a SHA-256 provenance
digest. New localization files carry an embedded UTC capture time, so copying
the JSON cannot make old perception look fresh. Its output deliberately stays
`fixture_source: true`: it contains the bottle plus static keepouts, not the
current dynamic RGB-D obstacle scene, so Gate C must still block execution.
Legacy JSON without the embedded time is rejected too. `--allow-stale` exists
only for explicit historical regression replay.

Poses in the scenario are **TCP** poses. The full rigid
`tcp_transform_from_ik_link` lives in the arm config. The checked-in transform
is currently a `0.1682 m` translation along `+Z`, but the 4×4 representation
preserves measured tool rotation when calibration supplies one.

## Offline checks

```bash
python3 -m pytest -q \
  mtc_ws/src/grabber_mtc_planner/test/test_plan_only_contract.py \
  test/shelf_dispenser/test_localization_to_mtc_scenario.py \
  test/shelf_dispenser/test_mtc_random_shelf_batch.py \
  test/shelf_dispenser/test_mtc_pick_contract.py
```

They prove the plan-only contract, the complete-branch semantics and the
scenario/arm-config consistency. They prove nothing about planning success —
only a real ROS 2 Humble build and a real plan-only run can.

## Synthetic shelf batch (still plan-only)

Generate identical randomized bottle layouts for both arms without a camera or
controller:

```bash
python3 scripts/mtc_random_shelf_batch.py \
  --output-dir /tmp/mtc_synthetic_batch \
  --count 3
```

In a ROS 2/MoveIt environment with an execution-disabled `move_group` and a
nonzero fixture joint state already publishing, add `--plan`. The summary
records solve counts and earliest failure stages per arm. Every generated case
has both `simulation_source: true` and `fixture_source: true`; the batch runner
rejects any other input, and the hardware execution bundle rejects its output.

## Gate C review (still no motion)

`gate_c_review.py` binds a plan result, bridge status, arm config and scenario
by SHA-256 and fails closed on stale/default state, unhealthy read-only bridge,
an uncalibrated selected arm, config/version mismatch, or a fixture scenario.
It never sends a trajectory and never reports `READY_FOR_EXECUTION`; its
strongest verdict is `READY_FOR_HUMAN_REVIEW`.

```bash
ros2 run grabber_mtc_planner gate_c_review.py \
  --result mtc_plan_result.json \
  --bridge-status bridge_status.json \
  --arms dual_rm75_arms.yaml \
  --scenario live_scenario.yaml \
  --out gate_c_review.json
```

A green review still requires a fresh start-state/collision recheck, explicit
human approval, and an operator at the hardware e-stop before any future
executor may move the robot.
