# Electronic fence profiles

The grasp state machine is environment-independent. Static geometry and allowed
TCP corridors live in `safety_profiles.json`; RGB-D contributes only temporary
obstacle voxels.

Each profile uses the calibrated/controller frame `right_controller_base`.
The fixed `T_moveit_from_profile` bridge converts it to MoveIt's
`platform_base_link` frame. A profile contains:

- `tcp_workspace`: the outer software envelope.
- `allowed_tcp_zones`: a union of safe home, transit, and task volumes.
- `keepout_boxes`: solid fixtures such as a table, shelf panels, shelf boards,
  walls, or the robot body.
- `clearance_m`: the extra distance applied during offline TCP checks.
- `verified_for_execution`: must remain `false` until every boundary has been
  measured and checked on the real robot.

### Tool-installation transform is a separate execution gate

For every right-arm profile, `tool_mount_calibration` is mandatory when
`verified_for_execution: true`. It records two full rigid transforms, not scalar
offsets:

- `T_link7_controller_flange`: controller flange expressed in MoveIt
  `r_link7`.
- `T_controller_flange_tcp`: physical bottle TCP expressed in controller
  flange.

The record must also include a real evidence id, UTC measurement time, and
position/orientation residuals. It is deliberately invalid to infer an
identity rotation from `tcp_z_m` or `moveit_link7_to_controller_flange_m`.
As a software admission ceiling, the recorded maximum residual must not exceed
5 mm or 1.0°. These bounds are not a substitute for the required same-joint
controller/MoveIt/physical-tool comparison.
Until the transform is populated and independently compared at identical joint
states against MoveIt `r_link7` FK and controller TCP/FK, keep both
`tool_mount_calibration.verified` and `verified_for_execution` false. The
checked-in `table_demo`, `shelf_template` and no-environment profiles
intentionally do so.

For a shelf, copy `shelf_template`, measure the shelf bottom/top/left/right/back
panels as keepout boxes, then define one home corridor and one or more bin
approach volumes. Select it with:

```bash
SAFETY_PROFILE=my_shelf PLAN_ONLY=1 scripts/start_task.sh
```

Only after plan-only validation and a low-speed supervised dry run should the
profile be marked `verified_for_execution: true`.

### Shelf panel keepout box ids and per-run adaptation

`shelf_bottom`/`shelf_top`/`shelf_back`/`shelf_left_panel`/`shelf_right_panel`
are not arbitrary names — `shelf_dispenser/shelf_model.py`'s `FACE_SPECS`
registry recognizes exactly these five ids and, every run, refits each one
against a fresh head-camera point cloud near the target the same way
`table_model.py` already does for `table_top` on `table_demo` (median across
multiple frames, in-plane extent only ever grows, hard abort if the measured
position drifts outside `shelf_fit_bound_tolerance_m` of the configured
value). Do not rename these ids without updating that registry, and do not
add a sixth ad-hoc face id expecting it to be adapted automatically — it will
silently pass through unmodified instead (harmless, but not what you want).

To get the initial numbers for a real shelf, run
`scripts/measure_shelf_geometry.py` on the robot (no arm motion — it only
starts the head camera). It prints a draft `keepout_boxes` fragment per face;
review and correct it by hand before pasting it into `shelf_template`. It
never writes `safety_profiles.json` itself, on purpose.

### Real dispensing vs. table_demo's place-back cycle

`table_demo`'s task flow is verify-and-replace: pick the bottle up, then put
it back at the same locked point (`RunOrchestrator._place_back`). A shelf/vending
deployment instead needs to *deliver* the bottle to an output/pickup point —
`RunOrchestrator._deliver_to_output`, selected via `task.DeliverMode.DISPENSE`
(CLI: `--task-mode ... --dispense`). It needs two additional profile fields
that `table_demo` does not set:

- `output_joints_deg`: 7-number joint target for the delivery transfer, same
  contract as `home_joints_deg`. Absent by default; `_deliver_to_output`
  fails closed (`SafetyAbort`) rather than guessing a delivery point.
- `output_visible_to_head_camera` (+ `output_point_base` when true): whether
  a real 3-D release check is possible at the output point. Most output
  points will *not* be in the head camera's field of view, so the default is
  `false` and release is judged from gripper feedback alone — this is
  intentionally weaker evidence than `table_demo`'s vision-confirmed release,
  logged as such rather than silently reusing the stronger claim.

Both fields are placeholders in the checked-in `shelf_template` until a real
output point is measured on site.

When a good route has already been demonstrated by teleoperation, record the
whole route instead of guessing a new global path:

```bash
python scripts/record_right_arm_guided_path.py
```

The recorder is read-only. The sampled joint/TCP path can be checked against
the same profile, reduced to safe gateway waypoints, and reused as the
environment-specific transit corridor. For shelves, keep a guided corridor per
bin or per row while retaining the same grasp state machine.

The RealMan SDK's named electronic-fence API is not used as a real-hardware
guard because its current documentation says that feature only takes effect in
controller simulation mode. On hardware, this demo uses MoveIt collision
objects plus SDK forward-kinematics validation of every dense trajectory point.
The hardware emergency stop remains required.
