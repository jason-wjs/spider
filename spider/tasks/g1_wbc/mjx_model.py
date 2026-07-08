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
    mjx_impl: str | None
    mjx_warp_naconmax: int | None
    mjx_warp_njmax: int | None
    mjx_model_options: dict[str, int]
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
    mjx_impl: str | None = None,
    mjx_warp_naconmax: int | None = None,
    mjx_warp_njmax: int | None = None,
    mjx_model_options: dict[str, int] | None = None,
) -> MjxModelBundle:
    """Build a CPU MuJoCo model bundle and optionally put it on MJX."""

    mjx_impl = _normalize_mjx_impl(mjx_impl)
    mjx_model_options = _normalize_model_options(mjx_model_options)
    profile = get_contact_profile(profile_name)
    if not profile.eligible_for_parity:
        raise ValueError(
            f"Contact profile {profile.name!r} cannot be bound to the WXY model bundle."
        )
    cpu_model = load_wbc_model(
        model_path,
        include_self_collision_sensors=not require_runtime,
        collision_profile=profile.name,
    )
    _apply_model_options(cpu_model, mjx_model_options)
    _assert_model_dims(cpu_model)
    maps = _build_name_maps(cpu_model)
    _assert_required_names(maps, profile)
    mjx_model = None
    if require_runtime:
        runtime = require_mjx_runtime()
        if mjx_impl is None:
            mjx_model = runtime.mjx.put_model(cpu_model)
        else:
            mjx_model = runtime.mjx.put_model(cpu_model, impl=mjx_impl)
        mjx_impl = _impl_name(getattr(mjx_model, "impl", mjx_impl))
    return MjxModelBundle(
        cpu_model=cpu_model,
        mjx_model=mjx_model,
        mjx_impl=mjx_impl,
        mjx_warp_naconmax=_positive_int_or_none(
            "mjx_warp_naconmax",
            mjx_warp_naconmax,
        ),
        mjx_warp_njmax=_positive_int_or_none("mjx_warp_njmax", mjx_warp_njmax),
        mjx_model_options=mjx_model_options,
        profile=profile,
        body_name_to_id=maps["body"],
        joint_name_to_id=maps["joint"],
        actuator_name_to_id=maps["actuator"],
        geom_name_to_id=maps["geom"],
    )


def make_mjx_data(bundle: MjxModelBundle, *, runtime):
    """Create an MJX Data object using the bundle's selected implementation."""

    if getattr(bundle, "mjx_model", None) is None:
        raise ValueError("MJX model bundle must include mjx_model")
    impl = _normalize_mjx_impl(getattr(bundle, "mjx_impl", None))
    if impl == "warp":
        kwargs: dict[str, int | str] = {"impl": "warp"}
        naconmax = getattr(bundle, "mjx_warp_naconmax", None)
        njmax = getattr(bundle, "mjx_warp_njmax", None)
        if naconmax is not None:
            kwargs["naconmax"] = int(naconmax)
        if njmax is not None:
            kwargs["njmax"] = int(njmax)
        return runtime.mjx.make_data(bundle.cpu_model, **kwargs)
    return runtime.mjx.make_data(bundle.mjx_model)


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
        _require_name(maps["geom"], geom_name, "geom")
    for floor_geom_name in profile.floor_geom_names:
        _require_name(maps["geom"], floor_geom_name, "geom")
    for geom_a, geom_b in profile.explicit_pair_names:
        _require_name(maps["geom"], geom_a, "geom")
        _require_name(maps["geom"], geom_b, "geom")


def _require_name(names: dict[str, int], name: str, kind: str) -> None:
    if name not in names:
        raise ValueError(f"Missing MJX {kind} name {name!r}.")


def _normalize_mjx_impl(value: str | None) -> str | None:
    if value is None:
        return None
    name = str(value).strip().lower()
    if name.startswith("impl."):
        name = name.split(".", 1)[1]
    if name not in {"jax", "warp"}:
        raise ValueError(f"Unsupported MJX impl {value!r}; expected 'jax' or 'warp'.")
    return name


def _impl_name(value: Any) -> str | None:
    if value is None:
        return None
    return _normalize_mjx_impl(str(value))


def _normalize_model_options(value: dict[str, int] | None) -> dict[str, int]:
    if value is None:
        return {}
    allowed = {"iterations", "ls_iterations"}
    normalized: dict[str, int] = {}
    for name, raw in value.items():
        if name not in allowed:
            raise ValueError(f"Unsupported MJX model option {name!r}.")
        normalized[name] = _positive_int_or_none(f"mjx_model_options.{name}", raw)
    return normalized


def _apply_model_options(model: mujoco.MjModel, options: dict[str, int]) -> None:
    for name, value in options.items():
        if not hasattr(model.opt, name):
            raise ValueError(f"MJX model option {name!r} is not available.")
        setattr(model.opt, name, int(value))


def _positive_int_or_none(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value
