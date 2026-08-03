import threading

import pytest

from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.environment_guard import LeftArmStabilityGuard


class Reader:
    def __init__(self):
        self.values = [0.0] * 7

    def joints_deg(self):
        return list(self.values)


def test_guard_rejects_left_arm_drift_and_requests_stop():
    reader = Reader()
    stop = threading.Event()
    guard = LeftArmStabilityGuard(
        left_reader=reader,
        stop_event=stop,
        tolerance_deg=0.8,
        poll_s=10.0,
    )
    assert guard.start() == [0.0] * 7
    reader.values[4] = 1.2

    with pytest.raises(SafetyAbort, match="左臂已偏离"):
        guard.check()

    # A synchronous boundary check raises to the caller; the background path
    # owns the asynchronous stop request.  Close must preserve the rejection.
    with pytest.raises(SafetyAbort, match="左臂已偏离"):
        guard.close()


def test_background_guard_sets_shared_stop_event_on_drift():
    reader = Reader()
    stop = threading.Event()
    guard = LeftArmStabilityGuard(
        left_reader=reader,
        stop_event=stop,
        tolerance_deg=0.8,
        poll_s=0.01,
    )
    guard.start()
    reader.values[0] = 2.0

    assert stop.wait(0.5)
    with pytest.raises(SafetyAbort, match="左臂已偏离"):
        guard.close()
