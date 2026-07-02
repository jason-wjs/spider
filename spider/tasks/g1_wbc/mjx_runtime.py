"""Lazy MJX runtime discovery for the G1 WBC backend."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class MjxRuntimeStatus:
    available: bool
    jax_available: bool
    jnp_available: bool
    mjx_available: bool
    jax_version: str | None
    mujoco_version: str | None
    mjx_module: str | None
    visible_devices: tuple[str, ...]
    error: str | None


@dataclass(frozen=True)
class MjxRuntime:
    jax: ModuleType
    jnp: ModuleType
    mjx: ModuleType
    status: MjxRuntimeStatus


def probe_mjx_runtime() -> MjxRuntimeStatus:
    """Inspect whether JAX and MuJoCo MJX are importable."""

    visible_devices = tuple(
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    )
    try:
        jax = importlib.import_module("jax")
    except Exception as exc:
        return MjxRuntimeStatus(
            available=False,
            jax_available=False,
            jnp_available=False,
            mjx_available=False,
            jax_version=None,
            mujoco_version=None,
            mjx_module=None,
            visible_devices=visible_devices,
            error=str(exc),
        )
    try:
        importlib.import_module("jax.numpy")
    except Exception as exc:
        return MjxRuntimeStatus(
            available=False,
            jax_available=True,
            jnp_available=False,
            mjx_available=False,
            jax_version=getattr(jax, "__version__", None),
            mujoco_version=None,
            mjx_module=None,
            visible_devices=visible_devices,
            error=str(exc),
        )
    try:
        mujoco = importlib.import_module("mujoco")
    except Exception as exc:
        return MjxRuntimeStatus(
            available=False,
            jax_available=True,
            jnp_available=True,
            mjx_available=False,
            jax_version=getattr(jax, "__version__", None),
            mujoco_version=None,
            mjx_module=None,
            visible_devices=visible_devices,
            error=str(exc),
        )
    mujoco_version = getattr(mujoco, "__version__", None)
    try:
        mjx = importlib.import_module("mujoco.mjx")
    except Exception as exc:
        return MjxRuntimeStatus(
            available=False,
            jax_available=True,
            jnp_available=True,
            mjx_available=False,
            jax_version=getattr(jax, "__version__", None),
            mujoco_version=mujoco_version,
            mjx_module=None,
            visible_devices=visible_devices,
            error=str(exc),
        )
    return MjxRuntimeStatus(
        available=True,
        jax_available=True,
        jnp_available=True,
        mjx_available=True,
        jax_version=getattr(jax, "__version__", None),
        mujoco_version=mujoco_version,
        mjx_module=getattr(mjx, "__name__", "mujoco.mjx"),
        visible_devices=visible_devices,
        error=None,
    )


def require_mjx_runtime() -> MjxRuntime:
    """Return imported MJX modules or raise an actionable backend error."""

    status = probe_mjx_runtime()
    if not status.available:
        raise RuntimeError(
            "--mpc-backend mjx requires importable jax and mujoco.mjx. "
            f"Runtime probe failed: {status.error}"
        )
    try:
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
        mjx = importlib.import_module("mujoco.mjx")
    except Exception as exc:
        raise RuntimeError(
            "--mpc-backend mjx requires importable jax and mujoco.mjx. "
            f"Runtime import failed after probe succeeded: {exc}"
        ) from exc
    return MjxRuntime(jax=jax, jnp=jnp, mjx=mjx, status=status)
