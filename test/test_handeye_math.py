import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from utils.handeye_calibrator import HandEyeCalibrator


def make_transform(rotvec, translation):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    transform[:3, 3] = translation
    return transform


class HandEyeMathTest(unittest.TestCase):
    def test_eye_in_hand_recovers_synthetic_ground_truth(self):
        rng = np.random.default_rng(20260710)
        base_to_ends = [
            make_transform(
                rng.normal(size=3) * 0.8,
                rng.uniform([-0.35, -0.35, 0.15], [0.35, 0.35, 0.75]),
            )
            for _ in range(18)
        ]
        truth_end_to_camera = make_transform(
            [0.22, -0.31, 0.17], [0.055, -0.025, 0.095]
        )
        truth_base_to_target = make_transform(
            [-0.35, 0.18, 0.28], [0.42, 0.08, 0.22]
        )
        target_to_cams = [
            np.linalg.inv(truth_end_to_camera)
            @ np.linalg.inv(base_to_end)
            @ truth_base_to_target
            for base_to_end in base_to_ends
        ]

        estimated = HandEyeCalibrator._solve_handeye(base_to_ends, target_to_cams)

        self.assertIsNotNone(estimated)
        np.testing.assert_allclose(estimated, truth_end_to_camera, atol=1e-9)

    def test_reflection_is_rejected(self):
        reflection = np.eye(4)
        reflection[2, 2] = -1.0
        self.assertFalse(HandEyeCalibrator._is_valid_rigid_transform(reflection))

    def test_charuco_detection_and_reprojection_gate(self):
        calibrator = HandEyeCalibrator(
            None,
            None,
            board_type="charuco",
            squares_x=9,
            squares_y=12,
            square_length=0.030,
            marker_length=0.0225,
            min_charuco_corners=12,
            max_reprojection_error_px=2.0,
        )
        if hasattr(calibrator.board, "generateImage"):
            image = calibrator.board.generateImage(
                (900, 1200), marginSize=40, borderBits=1
            )
        else:
            image = calibrator.board.draw(
                (900, 1200), marginSize=40, borderBits=1
            )
        calibrator.camera_matrix = np.array(
            [[1000.0, 0.0, 450.0], [0.0, 1000.0, 600.0], [0.0, 0.0, 1.0]]
        )
        calibrator.dist_coeffs = np.zeros(5)

        pose = calibrator._find_pattern_in_image(image)

        self.assertTrue(calibrator._is_valid_rigid_transform(pose))
        self.assertLess(calibrator.last_reprojection_error_px, 2.0)


if __name__ == "__main__":
    unittest.main()
