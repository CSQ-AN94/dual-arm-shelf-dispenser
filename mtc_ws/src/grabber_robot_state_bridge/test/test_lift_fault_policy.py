#!/usr/bin/env python3
"""Small regression check for measured lift position vs motion readiness."""

from __future__ import annotations

import pathlib
import sys

PACKAGE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE))

from grabber_robot_state_bridge.sdk_reader import (  # noqa: E402
    LiftSample,
    StateReadError,
    validate_lift_sample,
)


def main() -> None:
    faulted = LiftSample(
        height_mm=819,
        position_m=0.819,
        enabled=False,
        error_flag=512,
        monotonic=0.0,
    )
    assert not faulted.motion_ready

    try:
        validate_lift_sample(faulted)
    except StateReadError:
        pass
    else:
        raise AssertionError("faulted lift must fail closed by default")

    validate_lift_sample(faulted, allow_faulted_position=True)

    outside_model = LiftSample(
        height_mm=1001,
        position_m=1.001,
        enabled=True,
        error_flag=0,
        monotonic=0.0,
    )
    try:
        validate_lift_sample(outside_model, allow_faulted_position=True)
    except StateReadError:
        pass
    else:
        raise AssertionError("diagnostic override must not bypass RobotModel limits")

    print("lift fault policy OK")


if __name__ == "__main__":
    main()
