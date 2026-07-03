import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
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

    def test_mjx_backend_enables_physics_scan_without_extra_flag(self) -> None:
        self.assertTrue(
            evaluate._mjx_physics_scan_enabled(
                SimpleNamespace(mpc_backend="mjx", mjx_enable_scan=False)
            )
        )
        self.assertFalse(
            evaluate._mjx_physics_scan_enabled(
                SimpleNamespace(mpc_backend="mujoco_warp", mjx_enable_scan=False)
            )
        )

    def test_mpc_payload_requires_explicit_safety_metadata(self) -> None:
        metadata = {
            "accepted": True,
            "accepted_windows": 40,
            "used_baseline_fallback": False,
        }

        self.assertIs(evaluate._required_mpc_metadata(metadata, "accepted"), True)
        with self.assertRaisesRegex(ValueError, "accepted"):
            evaluate._required_mpc_metadata({}, "accepted")

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

    def test_generic_optimizer_rejects_legacy_only_mpc_knobs(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-optimizer",
            "generic",
            "--mpc-guided-candidate",
            "--mpc-acceptance-gate",
            "--mpc-guided-joint-gain",
            "0.50",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "legacy optimizer"):
            evaluate._validate_backend_args(args)

    def test_file_sha256_hashes_saved_command_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            command = Path(tmp_dir) / "mpc_command.npz"
            command.write_bytes(b"saved command bytes")

            digest = evaluate._file_sha256(command)

        self.assertEqual(
            digest,
            "368e0095ec392ef10cec152a60864c463232a071f6266307cb5924a2e2ad487e",
        )


if __name__ == "__main__":
    unittest.main()
