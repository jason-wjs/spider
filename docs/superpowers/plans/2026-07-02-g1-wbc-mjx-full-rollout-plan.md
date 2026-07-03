# G1 WBC MJX Full-Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a selectable MJX/JAX full-rollout backend for G1 WBC MPC that preserves the frozen sweetpoint baseline quality and reaches at least `12x` steady-state end-to-end MPC speedup on one GPU per motion.

**Architecture:** Keep MuJoCo-Warp as the default backend and add an explicit MJX backend path behind `--mpc-backend mjx`. Freeze baseline and acceptance semantics first, then build MJX model/contact, policy, observation, scoring, rollout, optimizer, replay, and report layers behind lazy JAX/MJX imports.

**Tech Stack:** Python 3.12, Torch, MuJoCo, MuJoCo-Warp, Warp, NumPy, pytest/unittest, optional JAX/MJX runtime, single-GPU CUDA profiling.

---

## Scope And Stop Rules

- Work only in `/data_team/junsong/model-based/spider_worktrees/g1-wbc-mjx-full-rollout`.
- Do not change the original `/data_team/junsong/model-based/spider` worktree.
- Keep `mujoco_warp` as the default backend for all existing commands.
- A run with `--mpc-backend mjx` must fail loudly when JAX or MJX is unavailable; it must never fall back to MuJoCo-Warp during MJX timing.
- One motion inference command must expose exactly one GPU through `CUDA_VISIBLE_DEVICES` or target one logical device such as `cuda:0`.
- Do not claim the milestone until `jump` and `walk` pass quality, replay, no-fallback, no-saturation, and H100 `>=12x` steady-state speed gates.

## File Structure

- Create `spider/tasks/g1_wbc/mjx_runtime.py`
  - Lazy imports for `jax`, `jax.numpy`, and `mujoco.mjx`.
  - Device visibility diagnostics and a structured `MjxRuntimeStatus`.
- Create `spider/tasks/g1_wbc/mjx_backend.py`
  - User-facing `run_g1_wbc_mjx_mpc(...)` entry point.
  - No silent fallback; unavailable runtime raises a deterministic error.
- Create `spider/tasks/g1_wbc/acceptance.py`
  - Baseline manifest schema checks, baseline eligibility, quality gates, speed gates, and formal pass/fail classification.
- Create `scripts/run_g1_wbc_stage0_baseline.py`
  - Runs the frozen sweetpoint command for `jump` and `walk`, seeds `0/1/2`, saves artifacts, and writes `baseline_manifest.json`.
- Modify `scripts/profile_g1_wbc_mpc_phases.py`
  - Add seed forwarding, `--save-rollout`, backend forwarding, warmup/steady timing fields, and command metadata.
- Modify `spider/tasks/g1_wbc/evaluate.py`
  - Add `--mpc-backend {mujoco_warp,mjx}` with default `mujoco_warp`.
  - Route only MPC methods to MJX when requested.
  - Add `accepted`, `accepted_windows`, and `used_baseline_fallback` payload fields for the MuJoCo-Warp sampling path.
- Create `spider/tasks/g1_wbc/mjx_contacts.py`
  - Contact-pair manifest for `wxy_parity`, `hgpt_track_reference`, and `hgpt_loco_reference`.
- Create `spider/tasks/g1_wbc/mjx_model.py`
  - In-memory WXY-to-MJX model preparation, index maps, contact capacity checks, and parity assertions.
- Create `spider/tasks/g1_wbc/mjx_policy.py`
  - Torch checkpoint to JAX actor parameter conversion and JAX MLP forward.
- Create `spider/tasks/g1_wbc/mjx_obs.py`
  - JAX observation construction and history buffer update.
- Create `spider/tasks/g1_wbc/mjx_scoring.py`
  - JAX score terms matching `compute_rollout_scores`.
- Create `spider/tasks/g1_wbc/mjx_optimizer.py`
  - One-window JIT optimizer owning sampling, interpolation, rollout, scoring, and weighted update.
- Create `scripts/run_g1_wbc_mjx_acceptance.py`
  - Runs MJX candidates against a frozen baseline manifest and writes `acceptance_report.json`.
- Create tests under `tests/tasks/g1_wbc/`
  - `test_mjx_runtime.py`
  - `test_mjx_backend_cli.py`
  - `test_acceptance.py`
  - `test_stage0_baseline_runner.py`
  - `test_mjx_contacts.py`
  - `test_mjx_model.py`
  - `test_mjx_policy.py`
  - `test_mjx_obs.py`
  - `test_mjx_scoring.py`
  - `test_mjx_optimizer.py`

## Task 1: Runtime Probe And Backend Guard

**Files:**

- Create: `spider/tasks/g1_wbc/mjx_runtime.py`
- Create: `spider/tasks/g1_wbc/mjx_backend.py`
- Modify: `spider/tasks/g1_wbc/evaluate.py`
- Create: `tests/tasks/g1_wbc/test_mjx_runtime.py`
- Create: `tests/tasks/g1_wbc/test_mjx_backend_cli.py`

- [ ] **Step 1: Write failing runtime tests**

Add `tests/tasks/g1_wbc/test_mjx_runtime.py`:

```python
from spider.tasks.g1_wbc import mjx_runtime


def test_runtime_status_reports_missing_modules_without_raising(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "jax" or name.startswith("mujoco.mjx"):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    status = mjx_runtime.probe_mjx_runtime()

    assert status.available is False
    assert status.jax_available is False
    assert status.mjx_available is False
    assert "jax" in status.error.lower()


def test_require_mjx_runtime_raises_clear_error_when_unavailable(monkeypatch):
    monkeypatch.setattr(
        mjx_runtime,
        "probe_mjx_runtime",
        lambda: mjx_runtime.MjxRuntimeStatus(
            available=False,
            jax_available=False,
            mjx_available=False,
            jax_version=None,
            mujoco_version="3.7.0",
            mjx_module=None,
            visible_devices=(),
            error="No module named 'jax'",
        ),
    )

    try:
        mjx_runtime.require_mjx_runtime()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("require_mjx_runtime() did not raise")

    assert "--mpc-backend mjx" in message
    assert "No module named 'jax'" in message
```

- [ ] **Step 2: Write failing CLI backend tests**

Add `tests/tasks/g1_wbc/test_mjx_backend_cli.py`:

```python
import pytest

from spider.tasks.g1_wbc import evaluate


def test_parse_args_defaults_to_mujoco_warp_backend(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
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
        ],
    )

    args = evaluate._parse_args()

    assert args.mpc_backend == "mujoco_warp"


def test_mjx_backend_rejects_non_mpc_method(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "no_mpc",
            "--mpc-backend",
            "mjx",
        ],
    )

    args = evaluate._parse_args()

    with pytest.raises(ValueError, match="requires an MPC method"):
        evaluate._validate_backend_args(args)
```

- [ ] **Step 3: Run tests and verify red**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest \
  tests/tasks/g1_wbc/test_mjx_runtime.py \
  tests/tasks/g1_wbc/test_mjx_backend_cli.py -q
```

Expected: fail because `mjx_runtime`, `mjx_backend`, and `mpc_backend` are not implemented.

- [ ] **Step 4: Implement runtime probe**

Create `spider/tasks/g1_wbc/mjx_runtime.py`:

```python
"""Lazy MJX runtime discovery for the G1 WBC backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import ModuleType

import mujoco


@dataclass(frozen=True)
class MjxRuntimeStatus:
    available: bool
    jax_available: bool
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
    visible_devices = tuple(
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    )
    mujoco_version = getattr(mujoco, "__version__", None)
    try:
        import jax  # type: ignore[import-not-found]
    except Exception as exc:
        return MjxRuntimeStatus(
            available=False,
            jax_available=False,
            mjx_available=False,
            jax_version=None,
            mujoco_version=mujoco_version,
            mjx_module=None,
            visible_devices=visible_devices,
            error=str(exc),
        )
    try:
        from mujoco import mjx  # type: ignore[attr-defined]
    except Exception as exc:
        return MjxRuntimeStatus(
            available=False,
            jax_available=True,
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
        mjx_available=True,
        jax_version=getattr(jax, "__version__", None),
        mujoco_version=mujoco_version,
        mjx_module=getattr(mjx, "__name__", "mujoco.mjx"),
        visible_devices=visible_devices,
        error=None,
    )


def require_mjx_runtime() -> MjxRuntime:
    status = probe_mjx_runtime()
    if not status.available:
        raise RuntimeError(
            "--mpc-backend mjx requires importable jax and mujoco.mjx. "
            f"Runtime probe failed: {status.error}"
        )
    import jax  # type: ignore[import-not-found]
    import jax.numpy as jnp  # type: ignore[import-not-found]
    from mujoco import mjx  # type: ignore[attr-defined]

    return MjxRuntime(jax=jax, jnp=jnp, mjx=mjx, status=status)
```

- [ ] **Step 5: Implement MJX backend guard**

Create `spider/tasks/g1_wbc/mjx_backend.py`:

```python
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
```

- [ ] **Step 6: Add CLI backend flag and validation**

Modify `spider/tasks/g1_wbc/evaluate.py` with these changes:

```python
MPC_METHODS = ("g1_wbc_ee", "g1_wbc_joint", "g1_wbc_joint_global")
```

```python
def _validate_backend_args(args: argparse.Namespace) -> None:
    if args.mpc_backend == "mjx" and args.method not in MPC_METHODS:
        raise ValueError("--mpc-backend mjx requires an MPC method.")
```

```python
parser.add_argument(
    "--mpc-backend",
    choices=("mujoco_warp", "mjx"),
    default="mujoco_warp",
    help="MPC rollout backend. MuJoCo-Warp remains the default.",
)
```

Call `_validate_backend_args(args)` immediately after `_parse_args()`. In the MPC branch, route MJX explicitly:

```python
if args.mpc_backend == "mjx":
    from spider.tasks.g1_wbc.mjx_backend import run_g1_wbc_mjx_mpc

    mpc_run = run_g1_wbc_mjx_mpc(
        spider_config=spider_config,
        motion=motion,
        actor=actor,
        rollout_config=config,
        execute_rollout_config=execute_config,
        method=args.method,
        reward_weights=reward_weights,
        total_steps=total_steps,
        seed=int(args.seed),
    )
else:
    mpc_run = run_g1_wbc_sampling_mpc(
        spider_config,
        task,
        total_steps=total_steps,
    )
```

- [ ] **Step 7: Run runtime and CLI tests**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest \
  tests/tasks/g1_wbc/test_mjx_runtime.py \
  tests/tasks/g1_wbc/test_mjx_backend_cli.py -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add \
  spider/tasks/g1_wbc/mjx_runtime.py \
  spider/tasks/g1_wbc/mjx_backend.py \
  spider/tasks/g1_wbc/evaluate.py \
  tests/tasks/g1_wbc/test_mjx_runtime.py \
  tests/tasks/g1_wbc/test_mjx_backend_cli.py
git commit -m "feat: add MJX backend guard"
```

## Task 2: Baseline And Acceptance Logic

**Files:**

- Create: `spider/tasks/g1_wbc/acceptance.py`
- Create: `tests/tasks/g1_wbc/test_acceptance.py`

- [ ] **Step 1: Write failing acceptance tests**

Add `tests/tasks/g1_wbc/test_acceptance.py`:

```python
from spider.tasks.g1_wbc.acceptance import (
    BASELINE_THRESHOLDS,
    MjxQualityPolicy,
    evaluate_baseline_group,
    evaluate_mjx_group,
    evaluate_speed_gate,
)


def _baseline_repeat(motion, seed, score, root, body, ee, contact, control, acc):
    return {
        "motion": motion,
        "seed": seed,
        "status": "ok",
        "num_steps": 800,
        "mpc_accepted": True,
        "accepted_windows": 40,
        "mpc_used_baseline_fallback": False,
        "wall_time_sec": 100.0 + seed,
        "metrics": {
            "success": True,
            "score": score,
            "root_pos_error_mean": root,
            "body_global_pos_error_mean": body,
            "ee_global_pos_error_mean": ee,
            "ee_local_pos_error_mean": 0.035,
            "contact_mismatch_rate": contact,
            "control_delta_mean": control,
            "joint_acc_mean": acc,
            "num_steps": 800,
        },
        "artifacts": {
            "metrics_json": "metrics.json",
            "rollout_npz": "rollout.npz",
            "mpc_command_npz": "mpc_command.npz",
        },
    }


def test_jump_baseline_group_passes_thresholds():
    rows = [
        _baseline_repeat("jump", 0, -2.05, 0.060, 0.070, 0.080, 0.34, 0.42, 210.0),
        _baseline_repeat("jump", 1, -2.08, 0.061, 0.071, 0.081, 0.35, 0.43, 211.0),
        _baseline_repeat("jump", 2, -2.10, 0.062, 0.072, 0.082, 0.35, 0.44, 212.0),
    ]

    result = evaluate_baseline_group("jump", rows)

    assert result.passed is True
    assert result.envelope["score"]["mean"] >= BASELINE_THRESHOLDS["jump"].score_mean_min
    assert result.promoted_seed == 0


def test_walk_baseline_fails_on_contact_mean():
    rows = [
        _baseline_repeat("walk", seed, -1.20, 0.070, 0.075, 0.078, 0.18, 0.23, 110.0)
        for seed in (0, 1, 2)
    ]

    result = evaluate_baseline_group("walk", rows)

    assert result.passed is False
    assert "contact_mismatch_rate_mean" in result.failures


def test_mjx_group_fails_on_fallback_even_when_metrics_are_good():
    baseline = evaluate_baseline_group(
        "walk",
        [
            _baseline_repeat("walk", seed, -1.20, 0.070, 0.075, 0.078, 0.14, 0.23, 110.0)
            for seed in (0, 1, 2)
        ],
    )
    mjx_rows = [
        {
            **_baseline_repeat("walk", seed, -1.19, 0.071, 0.076, 0.079, 0.14, 0.23, 111.0),
            "mpc_used_baseline_fallback": seed == 1,
            "contact_saturated": False,
        }
        for seed in (0, 1, 2)
    ]

    result = evaluate_mjx_group(
        "walk",
        mjx_rows,
        baseline.envelope,
        MjxQualityPolicy.for_motion("walk"),
    )

    assert result.passed is False
    assert "fallback" in result.failures


def test_speed_gate_uses_steady_state_wall_time_ratio():
    result = evaluate_speed_gate(
        baseline_wall_time_sec=120.0,
        mjx_steady_state_wall_time_sec=9.5,
        min_speedup=12.0,
    )

    assert result.passed is True
    assert result.speedup > 12.0
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_acceptance.py -q
```

Expected: fail because `acceptance.py` does not exist.

- [ ] **Step 3: Implement dataclasses and metrics helpers**

Create `spider/tasks/g1_wbc/acceptance.py` with concrete thresholds from the acceptance criteria:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Any


ERROR_METRICS = (
    "root_pos_error_mean",
    "body_global_pos_error_mean",
    "ee_global_pos_error_mean",
    "ee_local_pos_error_mean",
    "contact_mismatch_rate",
    "control_delta_mean",
    "joint_acc_mean",
)


@dataclass(frozen=True)
class BaselineThreshold:
    success_count_min: int
    score_mean_min: float
    metric_max: dict[str, float]


BASELINE_THRESHOLDS = {
    "jump": BaselineThreshold(
        success_count_min=2,
        score_mean_min=-2.12,
        metric_max={
            "root_pos_error_mean": 0.065,
            "body_global_pos_error_mean": 0.075,
            "ee_global_pos_error_mean": 0.085,
            "contact_mismatch_rate": 0.36,
            "control_delta_mean": 0.46,
            "joint_acc_mean": 225.0,
        },
    ),
    "walk": BaselineThreshold(
        success_count_min=3,
        score_mean_min=-1.27,
        metric_max={
            "root_pos_error_mean": 0.075,
            "body_global_pos_error_mean": 0.080,
            "ee_global_pos_error_mean": 0.082,
            "ee_local_pos_error_mean": 0.038,
            "contact_mismatch_rate": 0.155,
            "control_delta_mean": 0.25,
            "joint_acc_mean": 120.0,
        },
    ),
}


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class BaselineGroupResult(GateResult):
    envelope: dict[str, dict[str, float]]
    promoted_seed: int | None


@dataclass(frozen=True)
class MjxQualityPolicy:
    score_mean_allowance: float
    score_worst_allowance: float
    error_ratio_max: float
    contact_abs_allowance: float
    control_ratio_max: float

    @staticmethod
    def for_motion(motion: str) -> "MjxQualityPolicy":
        if motion == "jump":
            return MjxQualityPolicy(-0.08, -0.08, 1.10, 0.02, 1.15)
        if motion == "walk":
            return MjxQualityPolicy(-0.05, -0.05, 1.10, 0.02, 1.15)
        raise ValueError(f"Unsupported motion for MJX quality policy: {motion}")


@dataclass(frozen=True)
class SpeedGateResult:
    passed: bool
    speedup: float
    failures: tuple[str, ...]


def evaluate_baseline_group(motion: str, repeats: list[dict[str, Any]]) -> BaselineGroupResult:
    failures = _global_repeat_failures(repeats)
    threshold = BASELINE_THRESHOLDS[motion]
    if len(repeats) < 3:
        failures.append("repeat_count")
    success_count = sum(bool(_metrics(row).get("success")) for row in repeats)
    if success_count < threshold.success_count_min:
        failures.append("success_count")
    envelope = _build_envelope(repeats)
    if envelope.get("score", {}).get("mean", float("-inf")) < threshold.score_mean_min:
        failures.append("score_mean")
    for metric, upper in threshold.metric_max.items():
        if envelope.get(metric, {}).get("mean", float("inf")) > upper:
            failures.append(f"{metric}_mean")
    promoted_seed = _promoted_seed(repeats) if not failures else None
    return BaselineGroupResult(
        passed=not failures,
        failures=tuple(dict.fromkeys(failures)),
        envelope=envelope,
        promoted_seed=promoted_seed,
    )


def evaluate_mjx_group(
    motion: str,
    repeats: list[dict[str, Any]],
    baseline_envelope: dict[str, dict[str, float]],
    policy: MjxQualityPolicy,
) -> GateResult:
    failures = _global_repeat_failures(repeats)
    if any(bool(row.get("mpc_used_baseline_fallback")) for row in repeats):
        failures.append("fallback")
    if any(bool(row.get("contact_saturated")) for row in repeats):
        failures.append("contact_saturation")
    envelope = _build_envelope(repeats)
    score_mean_floor = baseline_envelope["score"]["mean"] + policy.score_mean_allowance
    if envelope["score"]["mean"] < score_mean_floor:
        failures.append("score_mean")
    for metric in ("root_pos_error_mean", "body_global_pos_error_mean", "ee_global_pos_error_mean"):
        if envelope[metric]["mean"] > baseline_envelope[metric]["mean"] * policy.error_ratio_max:
            failures.append(metric)
    if envelope["contact_mismatch_rate"]["mean"] > baseline_envelope["contact_mismatch_rate"]["mean"] + policy.contact_abs_allowance:
        failures.append("contact_mismatch_rate")
    for metric in ("control_delta_mean", "joint_acc_mean"):
        if envelope[metric]["mean"] > baseline_envelope[metric]["mean"] * policy.control_ratio_max:
            failures.append(metric)
    return GateResult(passed=not failures, failures=tuple(dict.fromkeys(failures)))


def evaluate_speed_gate(
    *,
    baseline_wall_time_sec: float,
    mjx_steady_state_wall_time_sec: float,
    min_speedup: float,
) -> SpeedGateResult:
    if mjx_steady_state_wall_time_sec <= 0.0:
        return SpeedGateResult(False, 0.0, ("mjx_wall_time",))
    speedup = baseline_wall_time_sec / mjx_steady_state_wall_time_sec
    failures = () if speedup >= min_speedup else ("speedup",)
    return SpeedGateResult(not failures, speedup, failures)
```

- [ ] **Step 4: Implement helper functions**

Append to `spider/tasks/g1_wbc/acceptance.py`:

```python
def _global_repeat_failures(repeats: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for row in repeats:
        if row.get("status") != "ok":
            failures.append("status")
        if int(row.get("num_steps", _metrics(row).get("num_steps", -1))) != 800:
            failures.append("num_steps")
        if not bool(row.get("mpc_accepted")):
            failures.append("mpc_accepted")
        if int(row.get("accepted_windows", -1)) != 40:
            failures.append("accepted_windows")
        if bool(row.get("mpc_used_baseline_fallback")):
            failures.append("baseline_fallback")
        artifacts = row.get("artifacts", {})
        for key in ("metrics_json", "rollout_npz", "mpc_command_npz"):
            if not artifacts.get(key):
                failures.append(key)
        for value in _metrics(row).values():
            if isinstance(value, int | float) and not math.isfinite(float(value)):
                failures.append("finite_metrics")
    return failures


def _metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        raise TypeError("repeat metrics must be a mapping")
    return metrics


def _metric_values(repeats: list[dict[str, Any]], metric: str) -> list[float]:
    return [float(_metrics(row)[metric]) for row in repeats if metric in _metrics(row)]


def _build_envelope(repeats: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    envelope: dict[str, dict[str, float]] = {}
    for metric in ("score", *ERROR_METRICS):
        values = _metric_values(repeats, metric)
        if not values:
            continue
        envelope[metric] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "median": median(values),
        }
    return envelope


def _promoted_seed(repeats: list[dict[str, Any]]) -> int | None:
    if not repeats:
        return None
    best = max(repeats, key=lambda row: float(_metrics(row).get("score", float("-inf"))))
    return int(best["seed"])
```

- [ ] **Step 5: Run acceptance tests**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_acceptance.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add spider/tasks/g1_wbc/acceptance.py tests/tasks/g1_wbc/test_acceptance.py
git commit -m "feat: add G1 WBC acceptance gates"
```

## Task 3: Stage 0 Baseline Runner And Profiling Payload

**Files:**

- Create: `scripts/run_g1_wbc_stage0_baseline.py`
- Modify: `scripts/profile_g1_wbc_mpc_phases.py`
- Create: `tests/tasks/g1_wbc/test_stage0_baseline_runner.py`

- [ ] **Step 1: Write failing runner tests**

Add `tests/tasks/g1_wbc/test_stage0_baseline_runner.py`:

```python
from pathlib import Path

from scripts import run_g1_wbc_stage0_baseline as runner


def test_stage0_builds_sweetpoint_commands(tmp_path):
    args = runner.parse_args(
        [
            "--jump-motion",
            "/motions/jump.npz",
            "--walk-motion",
            "/motions/walk.npz",
            "--checkpoint",
            "/ckpts/model_11800.pt",
            "--reward-weights",
            "/weights/v14.json",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    commands = runner.build_stage0_commands(args)

    assert len(commands) == 6
    assert all("--mpc-backend" in command.argv for command in commands)
    assert all("mujoco_warp" in command.argv for command in commands)
    assert all("--save-rollout" in command.argv for command in commands)
    assert {command.motion for command in commands} == {"jump", "walk"}
    assert {command.seed for command in commands} == {0, 1, 2}


def test_manifest_rejects_missing_artifacts(tmp_path):
    row = {
        "motion": "jump",
        "seed": 0,
        "status": "ok",
        "output_dir": str(tmp_path / "jump_seed0"),
        "metrics": {"success": True, "score": -2.0, "num_steps": 800},
        "mpc_accepted": True,
        "accepted_windows": 40,
        "mpc_used_baseline_fallback": False,
        "wall_time_sec": 1.0,
    }

    validated = runner.attach_artifact_paths(row)

    assert validated["artifacts"]["metrics_json"] is None
    assert validated["artifacts"]["rollout_npz"] is None
    assert validated["artifacts"]["mpc_command_npz"] is None
```

- [ ] **Step 2: Implement runner parser and command builder**

Create `scripts/run_g1_wbc_stage0_baseline.py`:

```python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SWEETPOINT_ARGS = (
    "--method", "g1_wbc_joint_global",
    "--mpc-backend", "mujoco_warp",
    "--max-steps", "800",
    "--mpc-samples", "512",
    "--mpc-iterations", "2",
    "--mpc-planning-horizon-steps", "40",
    "--mpc-control-steps", "20",
    "--mpc-sampling-mode", "knot",
    "--mpc-knot-count", "8",
    "--mpc-temperature", "0.7",
    "--mpc-root-pos-sigma", "0.04",
    "--mpc-root-rot-sigma", "0.10",
    "--mpc-joint-sigma", "0.18",
    "--mpc-command-reg-weight", "0.0",
    "--mpc-command-smooth-weight", "0.0",
    "--mpc-guided-candidate",
    "--mpc-acceptance-gate",
    "--save-rollout",
)


@dataclass(frozen=True)
class Stage0Command:
    motion: str
    seed: int
    output_dir: Path
    argv: list[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jump-motion", required=True)
    parser.add_argument("--walk-motion", required=True)
    parser.add_argument("--motion-type", default="isaaclab")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reward-weights", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_stage0_commands(args: argparse.Namespace) -> list[Stage0Command]:
    commands: list[Stage0Command] = []
    motions = {"jump": args.jump_motion, "walk": args.walk_motion}
    for motion, path in motions.items():
        for seed in (0, 1, 2):
            output_dir = args.output_dir / motion / f"seed_{seed}"
            argv = [
                args.python_executable,
                "-m",
                "spider.tasks.g1_wbc.evaluate",
                "--motion",
                str(path),
                "--motion-type",
                args.motion_type,
                "--checkpoint",
                str(args.checkpoint),
                "--device",
                str(args.device),
                "--output-dir",
                str(output_dir),
                "--mpc-reward-weights",
                str(args.reward_weights),
                "--seed",
                str(seed),
                *SWEETPOINT_ARGS,
            ]
            commands.append(Stage0Command(motion, seed, output_dir, argv))
    return commands


def attach_artifact_paths(row: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(row["output_dir"]))
    artifacts = {
        "metrics_json": output_dir / "metrics.json",
        "rollout_npz": output_dir / "rollout.npz",
        "mpc_command_npz": output_dir / "mpc_command.npz",
    }
    row = dict(row)
    row["artifacts"] = {
        key: str(path) if path.is_file() else None for key, path in artifacts.items()
    }
    return row
```

- [ ] **Step 3: Implement execution and manifest writing**

Append to `scripts/run_g1_wbc_stage0_baseline.py`:

```python
def run_command(command: Stage0Command) -> dict[str, Any]:
    command.output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command.argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    row: dict[str, Any] = {
        "motion": command.motion,
        "seed": command.seed,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "output_dir": str(command.output_dir),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    metrics_path = command.output_dir / "metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text())
        row["metrics"] = payload.get("metrics", {})
        mpc = payload.get("mpc", {})
        row["mpc_accepted"] = bool(mpc.get("accepted", True))
        row["accepted_windows"] = int(mpc.get("accepted_windows", mpc.get("num_windows", -1)))
        row["mpc_used_baseline_fallback"] = bool(mpc.get("used_baseline_fallback", False))
        row["num_steps"] = int(row["metrics"].get("num_steps", -1))
    return attach_artifact_paths(row)


def write_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    manifest = {
        "schema_version": 1,
        "baseline_name": "g1_wbc_joint_global_s512_i2_h40_c20_k8_v14",
        "motions": ("jump", "walk"),
        "seeds": (0, 1, 2),
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "baseline_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    commands = build_stage0_commands(args)
    if args.dry_run:
        rows = [
            attach_artifact_paths(
                {
                    "motion": command.motion,
                    "seed": command.seed,
                    "status": "planned",
                    "output_dir": str(command.output_dir),
                    "argv": command.argv,
                }
            )
            for command in commands
        ]
    else:
        rows = [run_command(command) for command in commands]
    write_manifest(args.output_dir, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Extend profiler payload**

Modify `scripts/profile_g1_wbc_mpc_phases.py` so parser accepts and forwards:

```python
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--mpc-backend", choices=("mujoco_warp", "mjx"), default="mujoco_warp")
parser.add_argument("--save-rollout", action="store_true")
```

When building the evaluate command, append:

```python
"--seed", str(args.seed),
"--mpc-backend", args.mpc_backend,
```

Append `"--save-rollout"` only when `args.save_rollout` is true. The timing JSON row must include:

```python
row["seed"] = int(args.seed)
row["mpc_backend"] = args.mpc_backend
row["steady_state_wall_time_sec"] = row["wall_time_sec"]
row["compile_init_wall_time_sec"] = None
```

- [ ] **Step 5: Run runner tests**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_stage0_baseline_runner.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add \
  scripts/run_g1_wbc_stage0_baseline.py \
  scripts/profile_g1_wbc_mpc_phases.py \
  tests/tasks/g1_wbc/test_stage0_baseline_runner.py
git commit -m "feat: add G1 WBC stage0 baseline runner"
```

## Task 4: MJX Contact And Model Bundle

**Files:**

- Create: `spider/tasks/g1_wbc/mjx_contacts.py`
- Create: `spider/tasks/g1_wbc/mjx_model.py`
- Create: `tests/tasks/g1_wbc/test_mjx_contacts.py`
- Create: `tests/tasks/g1_wbc/test_mjx_model.py`

- [ ] **Step 1: Write contact manifest tests**

Add tests that assert:

```python
from spider.tasks.g1_wbc.mjx_contacts import CONTACT_PROFILES, get_contact_profile


def test_contact_profiles_include_wxy_and_hgpt_references():
    assert set(CONTACT_PROFILES) == {
        "wxy_parity",
        "hgpt_track_reference",
        "hgpt_loco_reference",
    }


def test_wxy_parity_profile_names_floor_and_feet():
    profile = get_contact_profile("wxy_parity")

    assert profile.name == "wxy_parity"
    assert "floor" in profile.floor_geom_names
    assert profile.max_contact_points >= 128
    assert profile.max_geom_pairs >= 256
    assert any("left" in name or "l_" in name for name in profile.foot_body_names)
    assert any("right" in name or "r_" in name for name in profile.foot_body_names)
```

- [ ] **Step 2: Implement contact profiles**

Create `mjx_contacts.py` with immutable dataclasses and explicit profile names. The WXY profile must include the current seven-capsule foot semantics; the HGPT profiles are reference-only and must carry `eligible_for_parity=False`.

- [ ] **Step 3: Write model bundle tests**

Add tests that call `build_mjx_model_bundle(model_path, profile_name="wxy_parity")` and assert:

```python
assert bundle.profile.name == "wxy_parity"
assert bundle.cpu_model.nq == QPOS_DIM
assert bundle.cpu_model.nv == QVEL_DIM
assert len(bundle.body_name_to_id) >= 30
assert len(bundle.joint_name_to_id) >= 29
assert bundle.profile.eligible_for_parity is True
```

Use `pytest.importorskip("mujoco")` and skip the `mujoco.mjx` conversion portion when `probe_mjx_runtime().mjx_available` is false. CPU MuJoCo XML parsing remains required.

- [ ] **Step 4: Implement model bundle**

Create a `MjxModelBundle` dataclass containing:

```python
@dataclass(frozen=True)
class MjxModelBundle:
    cpu_model: mujoco.MjModel
    mjx_model: object | None
    profile: ContactProfile
    body_name_to_id: dict[str, int]
    joint_name_to_id: dict[str, int]
    actuator_name_to_id: dict[str, int]
```

`build_mjx_model_bundle(...)` must load the configured XML into an in-memory `mujoco.MjModel`, build name maps from MuJoCo APIs, call `mjx.put_model(cpu_model)` only when `require_runtime=True`, and reject non-parity profiles for milestone evaluation.

- [ ] **Step 5: Run model tests and commit**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest \
  tests/tasks/g1_wbc/test_mjx_contacts.py \
  tests/tasks/g1_wbc/test_mjx_model.py -q
```

Commit:

```bash
git add \
  spider/tasks/g1_wbc/mjx_contacts.py \
  spider/tasks/g1_wbc/mjx_model.py \
  tests/tasks/g1_wbc/test_mjx_contacts.py \
  tests/tasks/g1_wbc/test_mjx_model.py
git commit -m "feat: add MJX model bundle"
```

## Task 5: JAX Actor Conversion

**Files:**

- Create: `spider/tasks/g1_wbc/mjx_policy.py`
- Create: `tests/tasks/g1_wbc/test_mjx_policy.py`

- [ ] **Step 1: Write actor parity tests**

Use a synthetic `WbcActor` with deterministic weights. Compare `actor(torch_obs)` against `jax_actor_forward(convert_wbc_actor_to_jax(actor), np_obs)` on CPU for a batch of two observations. Assert max absolute error `<=1e-5`.

- [ ] **Step 2: Implement converter and forward**

`mjx_policy.py` must expose:

```python
@dataclass(frozen=True)
class JaxActorParams:
    obs_mean: object
    obs_std: object
    layers: tuple[tuple[object, object], ...]


def convert_wbc_actor_to_jax(actor: WbcActor, *, jnp) -> JaxActorParams:
    state = actor.state_dict()
    layers = []
    for index in range(0, len(actor.mlp), 2):
        linear = actor.mlp[index]
        layers.append(
            (
                jnp.asarray(linear.weight.detach().cpu().numpy().T),
                jnp.asarray(linear.bias.detach().cpu().numpy()),
            )
        )
    return JaxActorParams(
        obs_mean=jnp.asarray(state["obs_mean"].detach().cpu().numpy()),
        obs_std=jnp.asarray(state["obs_std"].detach().cpu().numpy()),
        layers=tuple(layers),
    )


def jax_actor_forward(params: JaxActorParams, obs, *, jnp):
    x = (obs - params.obs_mean) / (params.obs_std + 1.0e-2)
    for layer_index, (weight, bias) in enumerate(params.layers):
        x = x @ weight + bias
        if layer_index < len(params.layers) - 1:
            x = jnp.where(x > 0.0, x, jnp.expm1(x))
    return x
```

The MLP activation is ELU after every hidden layer and no activation after the output layer. Normalization must match Torch: `(obs - mean) / (std + 1.0e-2)`.

- [ ] **Step 3: Run tests and commit**

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_mjx_policy.py -q
git add spider/tasks/g1_wbc/mjx_policy.py tests/tasks/g1_wbc/test_mjx_policy.py
git commit -m "feat: add JAX WBC actor conversion"
```

## Task 6: JAX Observation Parity

**Files:**

- Create: `spider/tasks/g1_wbc/mjx_obs.py`
- Create: `tests/tasks/g1_wbc/test_mjx_obs.py`

- [ ] **Step 1: Write observation shape and history tests**

Build synthetic current-state arrays and command arrays with the exact G1 constants. Assert:

```python
obs.shape[-1] == OBS_DIM
history.shape[-2:] == (OBS_HISTORY_LENGTH, OBS_DIM)
```

The first update must backfill all history frames with the first observation; later updates must shift left and insert the latest frame.

- [ ] **Step 2: Implement observation API**

Expose:

```python
from dataclasses import dataclass
from typing import Mapping

from spider.tasks.g1_wbc.constants import ACTION_DIM, OBS_DIM, OBS_HISTORY_LENGTH


@dataclass(frozen=True)
class JaxObsState:
    history: object
    last_action: object


OBS_FIELD_ORDER = (
    "projected_gravity",
    "base_ang_vel",
    "joint_pos_error",
    "joint_vel",
    "command",
    "last_action",
)


def build_wbc_observation(fields: Mapping[str, object], *, jnp):
    parts = [jnp.ravel(jnp.asarray(fields[name])) for name in OBS_FIELD_ORDER]
    obs = jnp.concatenate(parts, axis=0)
    if int(obs.shape[0]) > OBS_DIM:
        raise ValueError(f"Observation has {obs.shape[0]} values, expected <= {OBS_DIM}")
    if int(obs.shape[0]) < OBS_DIM:
        obs = jnp.pad(obs, (0, OBS_DIM - int(obs.shape[0])))
    return obs


def update_obs_history(history, obs, *, initialized: bool, jnp):
    obs = jnp.asarray(obs)
    if initialized:
        return jnp.concatenate([history[1:], obs[None, :]], axis=0)
    return jnp.repeat(obs[None, :], OBS_HISTORY_LENGTH, axis=0)
```

The first green implementation validates synthetic arrays and shape/order invariants. Task 9 is blocked until a Torch parity test against `obs.py` is added and passes for a real rollout state.

- [ ] **Step 3: Run tests and commit**

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_mjx_obs.py -q
git add spider/tasks/g1_wbc/mjx_obs.py tests/tasks/g1_wbc/test_mjx_obs.py
git commit -m "feat: add JAX WBC observation helpers"
```

## Task 7: JAX Scoring Parity

**Files:**

- Create: `spider/tasks/g1_wbc/mjx_scoring.py`
- Create: `tests/tasks/g1_wbc/test_mjx_scoring.py`

- [ ] **Step 1: Write synthetic score tests**

Construct tiny synthetic rollout/reference tensors with known squared-error terms. Assert that root/body/EE/contact/control/joint-acc terms match NumPy expected values and that larger errors lower the total score.

- [ ] **Step 2: Implement streaming scorer**

Expose:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class JaxScoreWeights:
    terms: dict[str, float]


def score_step(accumulator, step_state, reference_state, weights: JaxScoreWeights, *, jnp):
    terms = dict(accumulator)
    root_error = jnp.mean((step_state["root_pos"] - reference_state["root_pos"]) ** 2)
    body_error = jnp.mean((step_state["body_pos"] - reference_state["body_pos"]) ** 2)
    ee_error = jnp.mean((step_state["ee_pos"] - reference_state["ee_pos"]) ** 2)
    contact_error = jnp.mean(jnp.abs(step_state["contact"] - reference_state["contact"]))
    control_delta = jnp.mean((step_state["control"] - step_state["prev_control"]) ** 2)
    joint_acc = jnp.mean((step_state["joint_vel"] - step_state["prev_joint_vel"]) ** 2)
    terms["root_pos_error_sum"] = terms.get("root_pos_error_sum", 0.0) + root_error
    terms["body_global_pos_error_sum"] = terms.get("body_global_pos_error_sum", 0.0) + body_error
    terms["ee_global_pos_error_sum"] = terms.get("ee_global_pos_error_sum", 0.0) + ee_error
    terms["contact_mismatch_sum"] = terms.get("contact_mismatch_sum", 0.0) + contact_error
    terms["control_delta_sum"] = terms.get("control_delta_sum", 0.0) + control_delta
    terms["joint_acc_sum"] = terms.get("joint_acc_sum", 0.0) + joint_acc
    terms["count"] = terms.get("count", 0.0) + 1.0
    total = (
        weights.terms.get("root_pos", 0.0) * root_error
        + weights.terms.get("body_global_pos", 0.0) * body_error
        + weights.terms.get("ee_global_pos", 0.0) * ee_error
        + weights.terms.get("contact", 0.0) * contact_error
        + weights.terms.get("control_delta", 0.0) * control_delta
        + weights.terms.get("joint_acc", 0.0) * joint_acc
    )
    terms["score_sum"] = terms.get("score_sum", 0.0) - total
    return terms


def finalize_score(accumulator, *, jnp):
    count = jnp.maximum(accumulator["count"], 1.0)
    return {
        "score": accumulator["score_sum"] / count,
        "root_pos_error_mean": accumulator["root_pos_error_sum"] / count,
        "body_global_pos_error_mean": accumulator["body_global_pos_error_sum"] / count,
        "ee_global_pos_error_mean": accumulator["ee_global_pos_error_sum"] / count,
        "contact_mismatch_rate": accumulator["contact_mismatch_sum"] / count,
        "control_delta_mean": accumulator["control_delta_sum"] / count,
        "joint_acc_mean": accumulator["joint_acc_sum"] / count,
    }
```

The metric names must match `compute_rollout_scores` so acceptance can compare MJX and replay metrics without translation.

- [ ] **Step 3: Run tests and commit**

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_mjx_scoring.py -q
git add spider/tasks/g1_wbc/mjx_scoring.py tests/tasks/g1_wbc/test_mjx_scoring.py
git commit -m "feat: add JAX WBC scoring helpers"
```

## Task 8: One-Window MJX Optimizer

**Files:**

- Create: `spider/tasks/g1_wbc/mjx_optimizer.py`
- Create: `tests/tasks/g1_wbc/test_mjx_optimizer.py`

- [ ] **Step 1: Write optimizer shape tests**

For deterministic fake rollout scores, assert:

```python
result.updated_controls.shape == controls.shape
result.execute_chunk.shape[0] == ctrl_steps + 1
result.info["best_score"] == expected_best
```

Add a test that `sample_residual_controls` returns identical samples for the same PRNG seed and different samples for different seeds.

- [ ] **Step 2: Implement sampling and weighted update**

Expose:

```python
@dataclass(frozen=True)
class JaxWindowOptimizerConfig:
    samples: int
    horizon_steps: int
    control_steps: int
    knot_count: int
    temperature: float
    root_pos_sigma: float
    root_rot_sigma: float
    joint_sigma: float


@dataclass(frozen=True)
class JaxWindowResult:
    updated_controls: object
    execute_chunk: object
    info: dict[str, float]


def optimize_window(config, state, controls, reference, actor_params, model_bundle, key, *, runtime):
    rollout_fn = state["rollout_fn"]
    samples = sample_residual_controls(config, controls, key, runtime=runtime)
    scores = rollout_fn(samples, reference, actor_params, model_bundle)
    best_index = runtime.jnp.argmax(scores)
    weights = runtime.jax.nn.softmax(scores / config.temperature)
    updated_controls = runtime.jnp.sum(samples * weights[:, None, None], axis=0)
    execute_chunk = updated_controls[: config.control_steps + 1]
    return JaxWindowResult(
        updated_controls=updated_controls,
        execute_chunk=execute_chunk,
        info={"best_score": float(scores[best_index])},
    )
```

The first green version accepts an injected fake rollout function for unit tests. The production rollout path must not be marked complete until it calls MJX physics inside `jax.lax.scan`.

- [ ] **Step 3: Run tests and commit**

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_mjx_optimizer.py -q
git add spider/tasks/g1_wbc/mjx_optimizer.py tests/tasks/g1_wbc/test_mjx_optimizer.py
git commit -m "feat: add MJX window optimizer skeleton"
```

## Task 9: Full MJX Backend Wiring

**Files:**

- Modify: `spider/tasks/g1_wbc/mjx_backend.py`
- Modify: `spider/tasks/g1_wbc/evaluate.py`
- Create: `tests/tasks/g1_wbc/test_mjx_backend_integration.py`

- [ ] **Step 1: Write backend contract tests**

Test that `run_g1_wbc_mjx_mpc(...)` returns a `G1WbcMpcRun`-compatible object with:

```python
assert result.metadata["backend"] == "mjx"
assert result.metadata["accepted"] is True
assert result.metadata["used_baseline_fallback"] is False
assert result.result.num_windows == 40
assert result.result.rollout.qpos.shape[0] == 801
```

Use fakes for runtime/model/optimizer in unit tests; keep real GPU tests behind `pytest.mark.gpu`.

- [ ] **Step 2: Implement backend orchestration**

The backend must:

1. Build `MjxModelBundle` with `profile_name="wxy_parity"`.
2. Convert the loaded Torch actor to `JaxActorParams`.
3. Build motion cache tensors once.
4. Iterate receding windows in Python.
5. Call compiled `optimize_window` once per window.
6. Execute the chosen chunk through MJX for the timed path.
7. Export selected qpos/qvel command trajectories.
8. Build a `G1WbcSpiderResult` compatible result.
9. Record `compile_init_wall_time_sec` and `steady_state_wall_time_sec` separately.

- [ ] **Step 3: Run tests and commit**

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_mjx_backend_integration.py -q
git add spider/tasks/g1_wbc/mjx_backend.py spider/tasks/g1_wbc/evaluate.py tests/tasks/g1_wbc/test_mjx_backend_integration.py
git commit -m "feat: wire MJX G1 WBC backend"
```

## Task 10: Replay Gate And Formal Acceptance Runner

**Files:**

- Create: `scripts/run_g1_wbc_mjx_acceptance.py`
- Create: `tests/tasks/g1_wbc/test_mjx_acceptance_runner.py`

- [ ] **Step 1: Write acceptance runner tests**

Assert that the runner:

```python
assert planned.backend == "mjx"
assert planned.replay_backend == "mujoco_warp"
assert planned.motion in {"jump", "walk"}
assert planned.seed in {0, 1, 2}
```

Also assert that a report with any replay failure returns process exit code `1`.

- [ ] **Step 2: Implement formal runner**

The runner must read `baseline_manifest.json`, run paired MJX seeds, replay each `mpc_command.npz` with `evaluate.py --method replay_command`, call `evaluate_mjx_group` and `evaluate_speed_gate`, and write:

```json
{
  "schema_version": 1,
  "backend": "mjx_canonical",
  "baseline_manifest": "/abs/path/baseline_manifest.json",
  "motion_results": {},
  "replay_results": {},
  "speed_results": {},
  "passed": false
}
```

- [ ] **Step 3: Run tests and commit**

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_mjx_acceptance_runner.py -q
git add scripts/run_g1_wbc_mjx_acceptance.py tests/tasks/g1_wbc/test_mjx_acceptance_runner.py
git commit -m "feat: add MJX acceptance runner"
```

## Task 11: GPU Verification Sequence

**Files:**

- No source changes unless a prior test exposes a defect.
- Write generated experiment outputs under a user-selected directory outside the repository, for example `/data_team/junsong/model-based/g1_wbc_mjx_runs/20260702_stage0`.

- [ ] **Step 1: Baseline manifest**

Run on one H100:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_stage0_baseline.py \
  --jump-motion /ABS/PATH/jump.npz \
  --walk-motion /ABS/PATH/walk.npz \
  --checkpoint /data_team/junsong/model-based/wbc_results/assets/checkpoints/model_8000.pt \
  --reward-weights /ABS/PATH/g1_wbc_reward_weights_method_specific_v14_20260612.json \
  --output-dir /data_team/junsong/model-based/g1_wbc_mjx_runs/stage0_baseline \
  --device cuda:0
```

- [ ] **Step 2: MJX formal acceptance**

Run on one H100:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_mjx_acceptance.py \
  --baseline-manifest /data_team/junsong/model-based/g1_wbc_mjx_runs/stage0_baseline/baseline_manifest.json \
  --output-dir /data_team/junsong/model-based/g1_wbc_mjx_runs/mjx_acceptance_h100 \
  --device cuda:0
```

- [ ] **Step 3: Pass declaration**

The milestone can be declared only when the generated report shows:

```text
jump quality passed=true
walk quality passed=true
jump replay passed=true
walk replay passed=true
jump speedup >= 12.0
walk speedup >= 12.0
single_gpu=true
fallback=false
contact_saturated=false
```

Commit only source and tests, not generated GPU artifacts.

## Self-Review Notes

- Spec coverage: Tasks 1-3 cover baseline freezing, runtime guard, and formal gates; Tasks 4-9 cover MJX model/contact/policy/obs/scoring/rollout/optimizer/backend; Task 10 covers replay/formal acceptance; Task 11 covers H100 single-GPU verification.
- Dependency gap: the current environment does not import `jax` or `mujoco.mjx`; Task 1 intentionally makes this a visible runtime status and blocks MJX claims until the dependency/runtime commit is made.
- Quality gap: Task 6 initially validates shape/history; Torch observation numerical parity must be added before Task 9 can be marked complete.
- Performance gap: Task 8 unit tests use fake rollout for deterministic coverage; the `>=12x` claim is reserved for Task 11 GPU acceptance output.
