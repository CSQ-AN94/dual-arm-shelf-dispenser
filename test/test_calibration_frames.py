import unittest

import numpy as np

from utils.calibration import Calibration
from utils.config import load_config


class CalibrationFramesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_all_configured_transforms_are_se3(self):
        calibration = self.config.calibration
        transforms = [
            calibration.T_end_right_to_camera_rightwrist,
            calibration.T_end_left_to_camera_leftwrist,
            calibration.T_base_right_to_camera_head,
            calibration.T_base_left_to_camera_head,
            calibration.T_base_right_to_base_left,
            calibration.T_base_left_to_base_right,
        ]
        for transform in transforms:
            self.assertEqual(transform.shape, (4, 4))
            np.testing.assert_allclose(transform[3], [0, 0, 0, 1], atol=1e-9)
            np.testing.assert_allclose(
                transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-8
            )
            self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1.0, places=8)

    def test_dual_arm_transforms_close_both_loops(self):
        c = self.config.calibration
        derived = c.T_base_right_to_camera_head @ np.linalg.inv(
            c.T_base_left_to_camera_head
        )
        np.testing.assert_allclose(derived, c.T_base_right_to_base_left, atol=1e-9)
        np.testing.assert_allclose(
            c.T_base_right_to_base_left @ c.T_base_left_to_base_right,
            np.eye(4), atol=1e-9,
        )

    def test_active_arm_compatibility_alias(self):
        c = self.config.calibration
        expected = c.wrist_extrinsic(self.config.connections.active_arm)
        np.testing.assert_array_equal(c.T_end_to_camera, expected)

    def test_any_camera_can_be_composed_to_either_arm_base(self):
        c = self.config.calibration
        np.testing.assert_array_equal(
            c.camera_to_arm_base("head", "left"),
            c.T_base_left_to_camera_head,
        )
        T_right_base_to_end = np.eye(4)
        T_right_base_to_end[:3, 3] = [0.3, -0.1, 0.4]
        expected = (
            c.T_base_left_to_base_right
            @ T_right_base_to_end
            @ c.T_end_right_to_camera_rightwrist
        )
        np.testing.assert_allclose(
            c.camera_to_arm_base(
                "right_wrist", "left", T_base_to_end=T_right_base_to_end
            ),
            expected,
            atol=1e-12,
        )

    def test_eye_in_hand_chain_uses_configured_extrinsic(self):
        c = self.config.calibration
        K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
        model = Calibration(c.T_end_right_to_camera_rightwrist, K, np.zeros(5))
        point = model.get_point_base(50.0, 50.0, 1.0, np.eye(4))
        expected = c.T_end_right_to_camera_rightwrist @ np.array([0, 0, 1, 1])
        np.testing.assert_allclose(point, expected[:3], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
