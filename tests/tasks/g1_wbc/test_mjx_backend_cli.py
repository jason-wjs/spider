import unittest
from unittest import mock

from spider.tasks.g1_wbc import evaluate


class MjxBackendCliTest(unittest.TestCase):
    def test_parse_args_defaults_to_mujoco_warp_backend(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-samples",
            "512",
            "--mpc-iterations",
            "2",
            "--mpc-planning-horizon-steps",
            "40",
            "--mpc-control-steps",
            "20",
            "--mpc-knot-count",
            "8",
            "--mpc-temperature",
            "0.7",
            "--mpc-root-pos-sigma",
            "0.04",
            "--mpc-root-rot-sigma",
            "0.10",
            "--mpc-joint-sigma",
            "0.18",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        self.assertEqual(args.mpc_backend, "mujoco_warp")
        self.assertFalse(args.mjx_enable_scan)

    def test_parse_args_accepts_explicit_mjx_scan_enable(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mjx-enable-scan",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        self.assertTrue(args.mjx_enable_scan)

    def test_mjx_backend_rejects_non_mpc_method(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "no_mpc",
            "--mpc-backend",
            "mjx",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "requires an MPC method"):
            evaluate._validate_backend_args(args)


if __name__ == "__main__":
    unittest.main()
