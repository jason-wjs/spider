"""Compare saved G1 WBC rollout contact forces with MuJoCo mj_contactForce."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from spider.tasks.g1_wbc.constants import (
    LEFT_FOOT_BODY_NAME,
    RIGHT_FOOT_BODY_NAME,
    WXY_G1_MODEL_PATH,
)
from spider.tasks.g1_wbc.rollout import (
    _geom_is_robot_collision,
    _resolve_actuator_ids_by_joint,
    _resolve_name_id,
    load_wbc_model,
)


def analyze_contact_force_oracle(
    rollout: str | Path,
    *,
    model_path: str | Path = WXY_G1_MODEL_PATH,
    collision_profile: str | None = None,
    mjx_model_options: Mapping[str, int] | None = None,
    frames: Sequence[int] | None = None,
    max_frames: int | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Replay rollout states through MuJoCo and compare mj_contactForce."""

    rollout_path = Path(rollout).expanduser().resolve()
    metadata = _read_rollout_metadata(rollout_path)
    collision_profile = (
        str(collision_profile)
        if collision_profile is not None
        else str(metadata.get("collision_profile", "wxy_parity"))
    )
    effective_model_options = dict(metadata.get("mjx_model_options", {}))
    if mjx_model_options is not None:
        effective_model_options.update(
            {str(name): int(value) for name, value in mjx_model_options.items()}
        )
    model = load_wbc_model(model_path, collision_profile=collision_profile)
    _apply_model_options(model, effective_model_options)
    groups = resolve_wbc_floor_contact_groups(model)
    qpos = load_rollout_array(rollout_path, "qpos")
    qvel = load_rollout_array(rollout_path, "qvel")
    raw_floor_force = _load_saved_floor_force(rollout_path)
    controls = _optional_rollout_array(rollout_path, "controls")
    frame_count = min(
        int(qpos.shape[0]),
        int(qvel.shape[0]),
        int(raw_floor_force.shape[0]),
    )
    selected_frames = _select_frames(frame_count, frames=frames, max_frames=max_frames)
    oracle_floor_force = _oracle_floor_force_frames(
        model,
        groups,
        qpos=qpos,
        qvel=qvel,
        controls=controls,
        frames=selected_frames,
    )
    raw_selected = raw_floor_force[selected_frames]
    return _summarize(
        rollout_path=rollout_path,
        model_path=Path(model_path).expanduser(),
        collision_profile=collision_profile,
        mjx_model_options=effective_model_options,
        frames=selected_frames,
        raw_floor_force=raw_selected,
        oracle_floor_force=oracle_floor_force,
        top_k=top_k,
    )


def contact_api_floor_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    floor_geom_ids: tuple[int, ...],
    left_foot_geom_ids: tuple[int, ...],
    right_foot_geom_ids: tuple[int, ...],
    other_robot_geom_ids: tuple[int, ...],
) -> np.ndarray:
    """Return left/right/other floor normal forces via MuJoCo contact API."""

    groups = (
        tuple(int(v) for v in left_foot_geom_ids),
        tuple(int(v) for v in right_foot_geom_ids),
        tuple(int(v) for v in other_robot_geom_ids),
    )
    floor_ids = set(int(v) for v in floor_geom_ids)
    force = np.zeros(3, dtype=np.float64)
    contact_force = np.zeros(6, dtype=np.float64)
    for contact_id in range(int(data.ncon)):
        contact = data.contact[contact_id]
        geom = tuple(int(v) for v in contact.geom)
        if geom[0] < 0 or geom[1] < 0:
            continue
        if float(contact.dist) > float(contact.includemargin) + 1.0e-5:
            continue
        if geom[0] not in floor_ids and geom[1] not in floor_ids:
            continue
        mujoco.mj_contactForce(model, data, contact_id, contact_force)
        normal = max(float(contact_force[0]), 0.0)
        if normal == 0.0:
            continue
        for index, geom_ids in enumerate(groups):
            if geom[0] in geom_ids or geom[1] in geom_ids:
                force[index] += normal
                break
    return force


def load_rollout_array(path: str | Path, name: str) -> np.ndarray:
    """Load a rollout array and squeeze the single-env dimension if present."""

    rollout_path = Path(path).expanduser()
    with np.load(rollout_path, allow_pickle=False) as data:
        if name not in data:
            raise KeyError(f"{rollout_path} is missing {name!r}")
        value = np.asarray(data[name], dtype=np.float64)
    if value.ndim >= 3 and int(value.shape[1]) == 1:
        value = value[:, 0]
    return value.reshape(int(value.shape[0]), -1)


def _read_rollout_metadata(path: Path) -> dict[str, Any]:
    metrics_path = path.parent / "metrics.json"
    if not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    mpc = payload.get("mpc", {}) if isinstance(payload, dict) else {}
    metadata: dict[str, Any] = {}
    if isinstance(mpc, dict):
        if "collision_profile" in mpc:
            metadata["collision_profile"] = str(mpc["collision_profile"])
        options = mpc.get("mjx_model_options")
        if isinstance(options, dict):
            metadata["mjx_model_options"] = {
                str(name): int(value)
                for name, value in options.items()
                if name in {"iterations", "ls_iterations"}
            }
    if "collision_profile" not in metadata and isinstance(payload, dict):
        if "collision_profile" in payload:
            metadata["collision_profile"] = str(payload["collision_profile"])
    return metadata


def _apply_model_options(
    model: mujoco.MjModel,
    options: Mapping[str, int],
) -> None:
    for name, value in options.items():
        if name == "iterations":
            model.opt.iterations = int(value)
        elif name == "ls_iterations":
            model.opt.ls_iterations = int(value)


def resolve_wbc_floor_contact_groups(
    model: mujoco.MjModel,
) -> dict[str, tuple[int, ...]]:
    """Resolve WBC floor, left-foot, right-foot, and other robot geom groups."""

    floor_ids = tuple(
        geom_id
        for geom_id in (
            _resolve_name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain"),
            _resolve_name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"),
        )
        if geom_id >= 0
    )
    if not floor_ids:
        raise ValueError("G1 WBC model is missing a terrain/floor geom.")
    left_ids = _body_geom_ids(model, LEFT_FOOT_BODY_NAME)
    right_ids = _body_geom_ids(model, RIGHT_FOOT_BODY_NAME)
    foot_ids = set(left_ids) | set(right_ids)
    other_ids = tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if geom_id not in foot_ids
        and geom_id not in floor_ids
        and _geom_is_robot_collision(model, geom_id)
    )
    return {
        "floor": floor_ids,
        "left": left_ids,
        "right": right_ids,
        "other": other_ids,
    }


def _body_geom_ids(model: mujoco.MjModel, body_name: str) -> tuple[int, ...]:
    body_id = _resolve_name_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"G1 WBC model is missing body {body_name!r}")
    return tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) == body_id
    )


def _optional_rollout_array(path: Path, name: str) -> np.ndarray | None:
    with np.load(path, allow_pickle=False) as data:
        if name not in data:
            return None
    return load_rollout_array(path, name)


def _load_saved_floor_force(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        has_floor = "floor_contact_force" in data
    if has_floor:
        return load_rollout_array(path, "floor_contact_force")
    contact_force = load_rollout_array(path, "contact_force")
    if int(contact_force.shape[1]) >= 3:
        return contact_force[:, :3]
    if int(contact_force.shape[1]) != 2:
        raise ValueError(
            f"Expected contact_force with 2 or 3 columns, got {contact_force.shape}"
        )
    other = np.zeros((int(contact_force.shape[0]), 1), dtype=np.float64)
    return np.concatenate([contact_force, other], axis=1)


def _select_frames(
    frame_count: int,
    *,
    frames: Sequence[int] | None,
    max_frames: int | None,
) -> np.ndarray:
    if frames:
        selected = sorted({int(frame) for frame in frames if 0 <= int(frame) < frame_count})
        if not selected:
            raise ValueError("No requested frames fall within rollout length.")
        return np.asarray(selected, dtype=np.int64)
    if max_frames is not None and int(max_frames) > 0 and int(max_frames) < frame_count:
        return np.linspace(0, frame_count - 1, num=int(max_frames), dtype=np.int64)
    return np.arange(frame_count, dtype=np.int64)


def _oracle_floor_force_frames(
    model: mujoco.MjModel,
    groups: dict[str, tuple[int, ...]],
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    controls: np.ndarray | None,
    frames: np.ndarray,
) -> np.ndarray:
    data = mujoco.MjData(model)
    actuator_ids = np.asarray(_resolve_actuator_ids_by_joint(model), dtype=np.int64)
    force = np.zeros((int(frames.shape[0]), 3), dtype=np.float64)
    for row, frame in enumerate(frames):
        data.qpos[:] = qpos[int(frame), : int(model.nq)]
        data.qvel[:] = qvel[int(frame), : int(model.nv)]
        if controls is not None and int(model.nu) > 0:
            control_frame = min(max(int(frame) - 1, 0), int(controls.shape[0]) - 1)
            data.ctrl[:] = _joint_order_to_model_ctrl(
                model,
                actuator_ids,
                controls[control_frame, : int(model.nu)],
            )
        elif int(model.nu) > 0:
            data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        force[row] = contact_api_floor_forces(
            model,
            data,
            floor_geom_ids=groups["floor"],
            left_foot_geom_ids=groups["left"],
            right_foot_geom_ids=groups["right"],
            other_robot_geom_ids=groups["other"],
        )
    return force


def _joint_order_to_model_ctrl(
    model: mujoco.MjModel,
    actuator_ids: np.ndarray,
    ctrl: np.ndarray,
) -> np.ndarray:
    if (
        int(actuator_ids.shape[0]) == int(model.nu)
        and np.array_equal(actuator_ids, np.arange(int(model.nu), dtype=np.int64))
    ):
        return np.asarray(ctrl, dtype=np.float64)
    model_ctrl = np.zeros(int(model.nu), dtype=np.float64)
    model_ctrl[actuator_ids] = np.asarray(ctrl, dtype=np.float64)
    return model_ctrl


def _summarize(
    *,
    rollout_path: Path,
    model_path: Path,
    collision_profile: str,
    mjx_model_options: Mapping[str, int],
    frames: np.ndarray,
    raw_floor_force: np.ndarray,
    oracle_floor_force: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    raw_foot = raw_floor_force[:, :2]
    oracle_foot = oracle_floor_force[:, :2]
    delta = raw_foot - oracle_foot
    abs_delta = np.abs(delta)
    worst_order = np.argsort(-np.max(abs_delta, axis=1))
    worst = [
        _frame_summary(
            frame=int(frames[index]),
            raw_floor_force=raw_floor_force[index],
            oracle_floor_force=oracle_floor_force[index],
        )
        for index in worst_order[: max(0, int(top_k))]
    ]
    return {
        "rollout": str(rollout_path),
        "model_path": str(model_path),
        "collision_profile": str(collision_profile),
        "mjx_model_options": dict(mjx_model_options),
        "evaluated_frame_count": int(frames.shape[0]),
        "metrics": {
            "raw_foot_force_mean": _mean(raw_foot),
            "oracle_foot_force_mean": _mean(oracle_foot),
            "raw_foot_force_peak": _max(raw_foot),
            "oracle_foot_force_peak": _max(oracle_foot),
            "mean_abs_delta": _mean(abs_delta),
            "max_abs_delta": _max(abs_delta),
            "max_raw_to_oracle_ratio": _max_ratio(raw_foot, oracle_foot),
        },
        "worst_frames": worst,
    }


def _frame_summary(
    *,
    frame: int,
    raw_floor_force: np.ndarray,
    oracle_floor_force: np.ndarray,
) -> dict[str, Any]:
    raw_foot = raw_floor_force[:2]
    oracle_foot = oracle_floor_force[:2]
    return {
        "frame": int(frame),
        "raw_floor_force": [float(v) for v in raw_floor_force],
        "oracle_floor_force": [float(v) for v in oracle_floor_force],
        "foot_force_delta": [float(v) for v in raw_foot - oracle_foot],
        "foot_force_ratio": [
            _ratio(float(raw), float(oracle))
            for raw, oracle in zip(raw_foot, oracle_foot)
        ],
    }


def _mean(value: np.ndarray) -> float:
    if value.size == 0:
        return math.nan
    return float(np.mean(value))


def _max(value: np.ndarray) -> float:
    if value.size == 0:
        return math.nan
    return float(np.max(value))


def _max_ratio(value: np.ndarray, baseline: np.ndarray) -> float:
    ratios = [
        _ratio(float(left), float(right))
        for left, right in zip(value.reshape(-1), baseline.reshape(-1))
    ]
    return max(ratios) if ratios else math.nan


def _ratio(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return 1.0 if value == 0.0 else math.inf
    return float(value) / float(baseline)


def _parse_frames(value: str | None) -> tuple[int, ...] | None:
    if value is None or value.strip() == "":
        return None
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = analyze_contact_force_oracle(
        args.rollout,
        model_path=args.model_path,
        collision_profile=args.collision_profile,
        mjx_model_options=_parse_model_options(args.mjx_model_options),
        frames=_parse_frames(args.frames),
        max_frames=args.max_frames,
        top_k=args.top_k,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=WXY_G1_MODEL_PATH)
    parser.add_argument("--collision-profile")
    parser.add_argument(
        "--mjx-model-options",
        help="Comma-separated opt overrides, e.g. iterations=4,ls_iterations=5.",
    )
    parser.add_argument("--frames")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _parse_model_options(value: str | None) -> dict[str, int] | None:
    if value is None or value.strip() == "":
        return None
    options: dict[str, int] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        name, raw = part.split("=", 1)
        name = name.strip()
        if name not in {"iterations", "ls_iterations"}:
            raise ValueError(f"Unsupported model option {name!r}")
        options[name] = int(raw)
    return options


if __name__ == "__main__":
    main()
