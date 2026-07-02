"""MJX model bundle helpers for the G1 WBC backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    QPOS_DIM,
    QVEL_DIM,
    WXY_G1_MODEL_PATH,
)
from spider.tasks.g1_wbc.mjx_contacts import ContactProfile, get_contact_profile
from spider.tasks.g1_wbc.mjx_runtime import require_mjx_runtime
from spider.tasks.g1_wbc.rollout import load_wbc_model


@dataclass(frozen=True)
class MjxModelBundle:
    cpu_model: mujoco.MjModel
    mjx_model: Any | None
    profile: ContactProfile
    body_name_to_id: dict[str, int]
    joint_name_to_id: dict[str, int]
    actuator_name_to_id: dict[str, int]
    geom_name_to_id: dict[str, int]


def build_mjx_model_bundle(
    model_path: str | Path = WXY_G1_MODEL_PATH,
    *,
    profile_name: str = "wxy_parity",
    require_runtime: bool = False,
) -> MjxModelBundle:
    """Build a CPU MuJoCo model bundle and optionally put it on MJX."""

    profile = get_contact_profile(profile_name)
    if require_runtime and not profile.eligible_for_parity:
        raise ValueError(
            f"Contact profile {profile.name!r} is not eligible for parity evaluation."
        )
    cpu_model = load_wbc_model(model_path)
    _assert_model_dims(cpu_model)
    maps = _build_name_maps(cpu_model)
    _assert_required_names(maps, profile)
    mjx_model = None
    if require_runtime:
        runtime = require_mjx_runtime()
        mjx_model = runtime.mjx.put_model(cpu_model)
    return MjxModelBundle(
        cpu_model=cpu_model,
        mjx_model=mjx_model,
        profile=profile,
        body_name_to_id=maps["body"],
        joint_name_to_id=maps["joint"],
        actuator_name_to_id=maps["actuator"],
        geom_name_to_id=maps["geom"],
    )


def assert_mjx_model_parity(bundle: MjxModelBundle) -> None:
    """Assert baseline dimensions and profile eligibility for first milestone use."""

    if not bundle.profile.eligible_for_parity:
        raise ValueError(f"Profile {bundle.profile.name!r} is not parity-eligible.")
    _assert_model_dims(bundle.cpu_model)
    _assert_required_names(
        {
            "body": bundle.body_name_to_id,
            "joint": bundle.joint_name_to_id,
            "actuator": bundle.actuator_name_to_id,
            "geom": bundle.geom_name_to_id,
        },
        bundle.profile,
    )


def _assert_model_dims(model: mujoco.MjModel) -> None:
    if int(model.nq) != QPOS_DIM:
        raise ValueError(f"Expected nq={QPOS_DIM}, got {model.nq}.")
    if int(model.nv) != QVEL_DIM:
        raise ValueError(f"Expected nv={QVEL_DIM}, got {model.nv}.")
    if int(model.nu) != ACTION_DIM:
        raise ValueError(f"Expected nu={ACTION_DIM}, got {model.nu}.")


def _build_name_maps(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    return {
        "body": _name_map(model, mujoco.mjtObj.mjOBJ_BODY, int(model.nbody)),
        "joint": _name_map(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.njnt)),
        "actuator": _name_map(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(model.nu)),
        "geom": _name_map(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.ngeom)),
    }


def _name_map(
    model: mujoco.MjModel,
    objtype: mujoco.mjtObj,
    count: int,
) -> dict[str, int]:
    names: dict[str, int] = {}
    for index in range(count):
        name = mujoco.mj_id2name(model, objtype, index)
        if name:
            names[name] = int(index)
    return names


def _assert_required_names(
    maps: dict[str, dict[str, int]],
    profile: ContactProfile,
) -> None:
    for body_name in MUJOCO_BODY_NAMES:
        _require_name(maps["body"], f"robot/{body_name}", "body")
    for joint_name in MUJOCO_JOINT_NAMES:
        _require_name(maps["joint"], f"robot/{joint_name}", "joint")
        _require_name(maps["actuator"], f"robot/{joint_name}", "actuator")
    for geom_name in profile.foot_collision_geom_names:
        if profile.eligible_for_parity:
            _require_name(maps["geom"], geom_name, "geom")
    for floor_geom_name in profile.floor_geom_names:
        if profile.eligible_for_parity:
            _require_name(maps["geom"], floor_geom_name, "geom")


def _require_name(names: dict[str, int], name: str, kind: str) -> None:
    if name not in names:
        raise ValueError(f"Missing MJX {kind} name {name!r}.")
