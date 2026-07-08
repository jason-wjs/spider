"""Diagnose MJX-vs-MuJoCo-Warp replay divergence from saved artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FRAMES = (0, 1, 2, 3, 4, 5, 10, 20, 40, 80, 120, 160, 200, 400, 800)
DEFAULT_THRESHOLDS = (1.0e-5, 1.0e-3, 1.0e-2, 1.0e-1)


def main() -> None:
    args = _parse_args()
    if args.acceptance_report is not None:
        mjx_rollout, replay_rollout, command_npz = _paths_from_acceptance_report(
            args.acceptance_report,
            motion=args.motion,
            seed=args.seed,
        )
    else:
        if args.mjx_rollout is None or args.replay_rollout is None:
            raise SystemExit(
                "--mjx-rollout and --replay-rollout are required without "
                "--acceptance-report."
            )
        mjx_rollout = args.mjx_rollout
        replay_rollout = args.replay_rollout
        command_npz = args.command_npz

    report = analyze_replay_parity(
        mjx_rollout,
        replay_rollout,
        command_npz=command_npz,
        frames=tuple(args.frames),
        thresholds=tuple(args.thresholds),
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)


def analyze_replay_parity(
    mjx_rollout: str | Path,
    replay_rollout: str | Path,
    *,
    command_npz: str | Path | None = None,
    frames: tuple[int, ...] = DEFAULT_FRAMES,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Return frame-wise divergence diagnostics for rollout artifacts."""

    mjx_path = Path(mjx_rollout).expanduser().resolve()
    replay_path = Path(replay_rollout).expanduser().resolve()
    with np.load(mjx_path, allow_pickle=False) as mjx_data, np.load(
        replay_path,
        allow_pickle=False,
    ) as replay_data:
        mjx = {name: np.asarray(mjx_data[name]) for name in mjx_data.files}
        replay = {name: np.asarray(replay_data[name]) for name in replay_data.files}

    report: dict[str, Any] = {
        "mjx_rollout": str(mjx_path),
        "replay_rollout": str(replay_path),
        "comparisons": {
            "replay_vs_mjx": _compare_artifact_pair(
                replay,
                mjx,
                frames=frames,
                thresholds=thresholds,
            ),
        },
    }

    if command_npz is not None:
        command_path = Path(command_npz).expanduser().resolve()
        with np.load(command_path, allow_pickle=False) as command_data:
            command = {
                name: np.asarray(command_data[name]) for name in command_data.files
            }
        report["command_npz"] = str(command_path)
        command_rollout = _command_as_rollout(command)
        report["comparisons"]["mjx_vs_command"] = _compare_artifact_pair(
            mjx,
            command_rollout,
            frames=frames,
            thresholds=thresholds,
        )
        report["comparisons"]["replay_vs_command"] = _compare_artifact_pair(
            replay,
            command_rollout,
            frames=frames,
            thresholds=thresholds,
        )

    return report


def _compare_artifact_pair(
    lhs: dict[str, np.ndarray],
    rhs: dict[str, np.ndarray],
    *,
    frames: tuple[int, ...],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for name, lhs_value, rhs_value in _comparison_arrays(lhs, rhs):
        comparisons[name] = _series_summary(
            _norm_series(lhs_value, rhs_value),
            frames=frames,
            thresholds=thresholds,
        )
    return comparisons


def _comparison_arrays(
    lhs: dict[str, np.ndarray],
    rhs: dict[str, np.ndarray],
):
    if "qpos" in lhs and "qpos" in rhs:
        yield "root_pos", _squeeze_env(lhs["qpos"])[..., :3], _squeeze_env(rhs["qpos"])[
            ..., :3
        ]
        yield "qpos", _squeeze_env(lhs["qpos"]), _squeeze_env(rhs["qpos"])
    for name in (
        "qvel",
        "actions",
        "controls",
        "contact_indicator",
        "contact_force",
        "floor_contact_indicator",
        "floor_contact_force",
    ):
        if name in lhs and name in rhs:
            yield name, _squeeze_env(lhs[name]), _squeeze_env(rhs[name])


def _command_as_rollout(command: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rollout: dict[str, np.ndarray] = {}
    if "command_qpos_trajectory" in command:
        rollout["qpos"] = command["command_qpos_trajectory"]
    elif "refined_qpos" in command:
        rollout["qpos"] = command["refined_qpos"]
    if "command_qvel_trajectory" in command:
        rollout["qvel"] = command["command_qvel_trajectory"]
    return rollout


def _squeeze_env(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim >= 3 and int(value.shape[1]) == 1:
        return value[:, 0]
    return value


def _norm_series(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    length = min(int(lhs.shape[0]), int(rhs.shape[0]))
    if length < 1:
        return np.zeros((0,), dtype=np.float64)
    lhs_flat = np.asarray(lhs[:length], dtype=np.float64).reshape(length, -1)
    rhs_flat = np.asarray(rhs[:length], dtype=np.float64).reshape(length, -1)
    return np.linalg.norm(lhs_flat - rhs_flat, axis=1)


def _series_summary(
    diff: np.ndarray,
    *,
    frames: tuple[int, ...],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    selected = {
        str(frame): float(diff[int(frame)])
        for frame in frames
        if 0 <= int(frame) < int(diff.shape[0])
    }
    return {
        "length": int(diff.shape[0]),
        "mean": float(np.mean(diff)) if diff.size else 0.0,
        "mean_first_20": float(np.mean(diff[: min(21, diff.shape[0])]))
        if diff.size
        else 0.0,
        "max": float(np.max(diff)) if diff.size else 0.0,
        "first_exceed": {
            _threshold_key(threshold): _first_exceed(diff, threshold)
            for threshold in thresholds
        },
        "frames": selected,
    }


def _first_exceed(diff: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(diff > float(threshold))
    if hits.size < 1:
        return None
    return int(hits[0])


def _threshold_key(value: float) -> str:
    return f"{float(value):.0e}"


def _paths_from_acceptance_report(
    path: str | Path,
    *,
    motion: str | None,
    seed: int | None,
) -> tuple[Path, Path, Path | None]:
    report_path = Path(path).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mjx_row = _select_row(report.get("mjx_rows", []), motion=motion, seed=seed)
    replay_row = _select_row(report.get("replay_rows", []), motion=motion, seed=seed)
    return (
        Path(mjx_row["artifacts"]["rollout_npz"]),
        Path(replay_row["artifacts"]["rollout_npz"]),
        Path(mjx_row["artifacts"]["mpc_command_npz"])
        if mjx_row.get("artifacts", {}).get("mpc_command_npz")
        else None,
    )


def _select_row(rows: list[dict[str, Any]], *, motion: str | None, seed: int | None):
    for row in rows:
        if motion is not None and row.get("motion") != motion:
            continue
        if seed is not None and int(row.get("seed", -1)) != int(seed):
            continue
        return row
    raise ValueError(f"No row matched motion={motion!r}, seed={seed!r}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--acceptance-report",
        type=Path,
        help="acceptance_report.json containing MJX and replay artifact paths.",
    )
    source.add_argument(
        "--mjx-rollout",
        type=Path,
        help="MJX rollout.npz artifact. Requires --replay-rollout.",
    )
    parser.add_argument("--replay-rollout", type=Path)
    parser.add_argument("--command-npz", type=Path)
    parser.add_argument("--motion")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--frames",
        type=int,
        nargs="*",
        default=list(DEFAULT_FRAMES),
        help="Frame indices to include in the summary.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=list(DEFAULT_THRESHOLDS),
        help="Norm thresholds used for first-exceed diagnostics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
