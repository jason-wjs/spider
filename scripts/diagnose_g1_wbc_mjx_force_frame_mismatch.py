"""Locate MJX-vs-replay contact-force spikes and nearby state divergence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


FORCE_HIGH_STATE_NEAR = "force_high_state_near"
FORCE_HIGH_STATE_DIVERGED = "force_high_state_diverged"
FORCE_NOMINAL = "force_nominal"
FOOT_NAMES = ("left", "right")


def analyze_force_frame_mismatch(
    mjx_rollout: str | Path,
    replay_rollout: str | Path,
    *,
    force_ratio_high: float = 3.0,
    force_delta_high: float = 100.0,
    qpos_near_threshold: float = 0.05,
) -> dict[str, Any]:
    """Return the worst per-foot MJX-over-replay force frame and state context."""

    mjx_path = Path(mjx_rollout).expanduser().resolve()
    replay_path = Path(replay_rollout).expanduser().resolve()
    mjx = _load_npz(mjx_path)
    replay = _load_npz(replay_path)
    feet = {
        name: _foot_force_row(
            mjx,
            replay,
            foot_index=index,
            force_ratio_high=float(force_ratio_high),
            force_delta_high=float(force_delta_high),
            qpos_near_threshold=float(qpos_near_threshold),
        )
        for index, name in enumerate(FOOT_NAMES)
    }
    return {
        "mjx_rollout": str(mjx_path),
        "replay_rollout": str(replay_path),
        "thresholds": {
            "force_ratio_high": float(force_ratio_high),
            "force_delta_high": float(force_delta_high),
            "qpos_near_threshold": float(qpos_near_threshold),
        },
        "feet": feet,
        "classification_counts": _classification_counts(feet.values()),
    }


def _foot_force_row(
    mjx: dict[str, np.ndarray],
    replay: dict[str, np.ndarray],
    *,
    foot_index: int,
    force_ratio_high: float,
    force_delta_high: float,
    qpos_near_threshold: float,
) -> dict[str, Any]:
    mjx_force = _series(mjx, "contact_force")[:, int(foot_index)]
    replay_force = _series(replay, "contact_force")[:, int(foot_index)]
    length = min(len(mjx_force), len(replay_force))
    mjx_force = mjx_force[:length]
    replay_force = replay_force[:length]
    delta = mjx_force - replay_force
    frame = int(np.argmax(delta)) if delta.size else 0
    qpos_norm = _frame_norm(mjx, replay, "qpos", frame)
    qvel_norm = _frame_norm(mjx, replay, "qvel", frame)
    mjx_contact = _contact_value(mjx, frame, foot_index)
    replay_contact = _contact_value(replay, frame, foot_index)
    ratio = _ratio(float(mjx_force[frame]), float(replay_force[frame]))
    force_delta = float(delta[frame])
    same_contact = bool(abs(mjx_contact - replay_contact) <= 0.5)
    high_force = ratio >= force_ratio_high or force_delta >= force_delta_high
    near_state = qpos_norm <= qpos_near_threshold and same_contact
    if high_force and near_state:
        classification = FORCE_HIGH_STATE_NEAR
    elif high_force:
        classification = FORCE_HIGH_STATE_DIVERGED
    else:
        classification = FORCE_NOMINAL
    return {
        "frame": frame,
        "classification": classification,
        "mjx_force": float(mjx_force[frame]),
        "replay_force": float(replay_force[frame]),
        "force_delta": force_delta,
        "force_ratio": ratio,
        "qpos_norm": qpos_norm,
        "qvel_norm": qvel_norm,
        "mjx_contact": mjx_contact,
        "replay_contact": replay_contact,
        "same_contact": same_contact,
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _series(payload: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in payload:
        raise KeyError(f"rollout is missing {name!r}")
    value = np.asarray(payload[name], dtype=np.float64)
    if value.ndim >= 3 and int(value.shape[1]) == 1:
        value = value[:, 0]
    return value.reshape(int(value.shape[0]), -1)


def _frame_norm(
    mjx: dict[str, np.ndarray],
    replay: dict[str, np.ndarray],
    name: str,
    frame: int,
) -> float:
    left = _series(mjx, name)
    right = _series(replay, name)
    if frame >= min(int(left.shape[0]), int(right.shape[0])):
        return math.nan
    return float(np.linalg.norm(left[frame] - right[frame]))


def _contact_value(
    payload: dict[str, np.ndarray],
    frame: int,
    foot_index: int,
) -> float:
    contact = _series(payload, "contact_indicator")
    if frame >= int(contact.shape[0]) or foot_index >= int(contact.shape[1]):
        return math.nan
    return float(contact[frame, int(foot_index)])


def _ratio(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return 1.0 if value == 0.0 else math.inf
    return float(value) / float(baseline)


def _classification_counts(rows) -> dict[str, int]:
    counts = {
        FORCE_HIGH_STATE_NEAR: 0,
        FORCE_HIGH_STATE_DIVERGED: 0,
        FORCE_NOMINAL: 0,
    }
    for row in rows:
        classification = str(row.get("classification"))
        counts[classification] = counts.get(classification, 0) + 1
    return counts


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = analyze_force_frame_mismatch(
        args.mjx_rollout,
        args.replay_rollout,
        force_ratio_high=args.force_ratio_high,
        force_delta_high=args.force_delta_high,
        qpos_near_threshold=args.qpos_near_threshold,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjx-rollout", type=Path, required=True)
    parser.add_argument("--replay-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force-ratio-high", type=float, default=3.0)
    parser.add_argument("--force-delta-high", type=float, default=100.0)
    parser.add_argument("--qpos-near-threshold", type=float, default=0.05)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
