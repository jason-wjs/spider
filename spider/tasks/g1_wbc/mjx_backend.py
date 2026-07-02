"""MJX backend entry point for G1 WBC sampled MPC."""

from __future__ import annotations

from spider.tasks.g1_wbc.mjx_runtime import require_mjx_runtime


def run_g1_wbc_mjx_mpc(*args, **kwargs):
    """Run the MJX full-rollout backend or fail before touching Warp state."""

    require_mjx_runtime()
    raise NotImplementedError(
        "MJX runtime is available, but the full-rollout optimizer has not "
        "been wired yet. Keep using --mpc-backend mujoco_warp for production."
    )
