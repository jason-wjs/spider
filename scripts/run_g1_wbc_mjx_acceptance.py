#!/usr/bin/env python3
"""Run formal G1 WBC MJX acceptance against a frozen baseline manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Mapping

import numpy as np

from spider.tasks.g1_wbc.acceptance import (
    MjxQualityPolicy,
    PRIMARY_ERROR_METRICS,
    REQUIRED_ARTIFACT_FIELDS,
    SpeedGateResult,
    evaluate_baseline_group,
    evaluate_mjx_group,
    evaluate_speed_gate,
)
from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    MUJOCO_BODY_NAMES,
    POLICY_DT,
    QPOS_DIM,
    QVEL_DIM,
)

SPIDER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON_EXECUTABLE = (
    SPIDER_ROOT / ".venv" / "bin" / "python"
    if (SPIDER_ROOT / ".venv" / "bin" / "python").exists()
    else Path(sys.executable)
)
MOTIONS = ("jump", "walk")
SEEDS = (0, 1, 2)
MIN_SPEEDUP = 12.0
MIN_REALTIME_FACTOR = 1.0
DEFAULT_REQUIRED_GPU_NAME_FRAGMENT = "H100"
ARTIFACT_FRESHNESS_TOLERANCE_NS = 2_000_000_000
FORMAL_BASELINE_NAME = "g1_wbc_stage0_mujoco_warp_sweetpoint"
CONTACT_FORCE_SEMANTICS = "pyramidal_contact_normal_v1"
TARGET_H100_SPEEDUP = "h100_speedup"
COMMAND_QVEL_CONSISTENCY_ATOL = 5.0e-5
COMMAND_QVEL_CONSISTENCY_RTOL = 1.0e-5
TARGET_4090_REALTIME = "4090_realtime"
ACCEPTANCE_REPORT_NAME = "acceptance_report.json"
ACCEPTANCE_PARTIAL_REPORT_NAME = "acceptance_report.partial.json"
ACCEPTANCE_ROW_SIDECAR_NAME = "acceptance_row.json"
STAGE0_RUNNER_PROVENANCE_FILENAME = "stage0_runner_provenance.json"
REQUIRED_INPUT_SHA256_FIELDS = (
    "jump_motion",
    "walk_motion",
    "checkpoint",
    "reward_weights",
)
FORMAL_STAGE0_ARG_VALUES = {
    "--motion-type": "isaaclab",
    "--method": "g1_wbc_joint_global",
    "--mpc-backend": "mujoco_warp",
    "--mpc-optimizer": "legacy",
    "--mpc-preset": "aggressive",
    "--max-steps": "800",
    "--mpc-samples": "512",
    "--mpc-iterations": "2",
    "--mpc-planning-horizon-steps": "40",
    "--mpc-control-steps": "20",
    "--mpc-sampling-mode": "knot",
    "--mpc-knot-count": "8",
    "--mpc-elite-frac": "0.125",
    "--mpc-temperature": "0.7",
    "--mpc-root-pos-sigma": "0.04",
    "--mpc-root-rot-sigma": "0.10",
    "--mpc-joint-sigma": "0.18",
    "--mpc-sigma-decay": "0.75",
    "--mpc-smooth-passes": "0",
    "--mpc-command-reg-weight": "0.0",
    "--mpc-command-smooth-weight": "0.0",
    "--mpc-guided-root-pos-gain": "0.50",
    "--mpc-guided-root-rot-gain": "0.50",
    "--mpc-guided-joint-gain": "0.50",
    "--mpc-guided-root-pos-clip": "0.05",
    "--mpc-guided-root-rot-clip": "0.12",
    "--mpc-guided-joint-clip": "0.35",
    "--mpc-warm-start-source": "best",
    "--mpc-warm-start-decay": "1.0",
    "--nconmax-per-env": "512",
    "--njmax-per-env": "2048",
}
FORMAL_STAGE0_FLAGS = (
    "--save-rollout",
    "--mpc-guided-candidate",
    "--mpc-acceptance-gate",
    "--no-mpc-warm-start",
)
FORMAL_STAGE0_DYNAMIC_ARG_FLAGS = (
    "--motion",
    "--checkpoint",
    "--device",
    "--output-dir",
    "--seed",
    "--mpc-reward-weights",
)
FORMAL_STAGE0_FORBIDDEN_FLAGS = (
    "--no-mpc-guided-candidate",
    "--no-mpc-acceptance-gate",
    "--mpc-warm-start",
)
LEGACY_ONLY_MJX_DROP_ARG_VALUES = (
    "--mpc-preset",
    "--mpc-sampling-mode",
    "--mpc-smooth-passes",
    "--mpc-command-reg-weight",
    "--mpc-command-smooth-weight",
    "--mpc-guided-root-pos-gain",
    "--mpc-guided-root-rot-gain",
    "--mpc-guided-joint-gain",
    "--mpc-guided-root-pos-clip",
    "--mpc-guided-root-rot-clip",
    "--mpc-guided-joint-clip",
    "--mpc-warm-start-source",
    "--mpc-warm-start-decay",
)
LEGACY_ONLY_MJX_DROP_FLAGS = (
    "--mpc-guided-candidate",
    "--no-mpc-guided-candidate",
    "--mpc-acceptance-gate",
    "--no-mpc-acceptance-gate",
)
MJX_GENERIC_ARG_VALUES = {
    "--mpc-first-ctrl-noise-scale": "1.0",
    "--mpc-last-ctrl-noise-scale": "1.0",
    "--mpc-final-noise-scale": "1.0",
}
MJX_MPC_NUMERIC_OVERRIDE_FLAGS = {
    "mjx_mpc_root_pos_sigma": "--mpc-root-pos-sigma",
    "mjx_mpc_root_rot_sigma": "--mpc-root-rot-sigma",
    "mjx_mpc_joint_sigma": "--mpc-joint-sigma",
    "mjx_mpc_first_ctrl_noise_scale": "--mpc-first-ctrl-noise-scale",
    "mjx_mpc_last_ctrl_noise_scale": "--mpc-last-ctrl-noise-scale",
    "mjx_mpc_final_noise_scale": "--mpc-final-noise-scale",
    "mjx_mpc_sigma_decay": "--mpc-sigma-decay",
}
MJX_CONTACT_SATURATION_FIELDS = (
    "contact_saturated",
    "max_contact_points_saturated",
    "max_geom_pairs_saturated",
)
MJX_CONTACT_COUNT_FIELDS = (
    "max_contact_points",
    "max_geom_pairs",
    "contact_pair_count",
    "active_contact_count",
)


@dataclass(frozen=True)
class PlannedAcceptanceRun:
    motion: str
    seed: int
    backend: str
    replay_backend: str
    output_dir: str
    replay_output_dir: str
    mjx_argv: list[str]
    replay_argv: list[str]
    mjx_command_text: str
    replay_command_text: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-executable", default=str(DEFAULT_PYTHON_EXECUTABLE))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--target",
        choices=(TARGET_H100_SPEEDUP, TARGET_4090_REALTIME),
        default=TARGET_H100_SPEEDUP,
        help=(
            "Formal acceptance target. The H100 target gates on baseline-relative "
            "speedup; the 4090 target gates on both baseline-relative speedup "
            "and real-time factor."
        ),
    )
    parser.add_argument("--min-speedup", type=float, default=MIN_SPEEDUP)
    parser.add_argument(
        "--min-realtime-factor",
        type=float,
        default=MIN_REALTIME_FACTOR,
        help="Minimum motion-duration / MJX steady-state wall-time for the 4090 target.",
    )
    parser.add_argument(
        "--required-gpu-name-fragment",
        default=DEFAULT_REQUIRED_GPU_NAME_FRAGMENT,
        help=(
            "Substring required in baseline, MJX, and replay runtime GPU names "
            "for this acceptance milestone. Use an empty value to disable the "
            "model-name check."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-existing-ok",
        action="store_true",
        help=(
            "Reuse previously completed ok MJX/replay rows from this output-dir "
            "when command, metrics provenance, artifact hashes, and schema match."
        ),
    )
    parser.add_argument(
        "--collision-profile",
        default=None,
        help=(
            "Override the WXY collision profile for both MJX optimization and "
            "replay validation rows. Omit to preserve the baseline row setting."
        ),
    )
    parser.add_argument(
        "--mjx-guided-candidate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable MJX-generated guided candidate controls in the generic MJX "
            "acceptance run. The replay validation run never receives this flag."
        ),
    )
    parser.add_argument(
        "--mjx-guided-candidate-period",
        type=int,
        default=None,
        help=(
            "Diagnostic: when guided candidate is enabled, generate it every N "
            "MPC windows to reduce guided no-MPC rollout overhead."
        ),
    )
    parser.add_argument(
        "--mjx-guided-candidate-period-override",
        action="append",
        default=[],
        metavar="MOTION:SEED:VALUE",
        help=(
            "Enable guided candidate and override its period for one MJX row, "
            "e.g. jump:0:4. Repeat for multiple rows."
        ),
    )
    parser.add_argument(
        "--mjx-min-score-improvement",
        type=float,
        default=None,
        help=(
            "Diagnostic: require this score improvement before accepting a "
            "non-current MJX optimizer candidate. Defaults to evaluate.py's "
            "legacy 1e-9 when omitted."
        ),
    )
    parser.add_argument(
        "--mjx-min-top-score-gap",
        type=float,
        default=None,
        help=(
            "Diagnostic: require this best-vs-second-best score gap before "
            "accepting a non-current MJX optimizer candidate. Defaults to "
            "evaluate.py's 0.0 when omitted."
        ),
    )
    parser.add_argument(
        "--mjx-cem-update-min-top-score-gap",
        type=float,
        default=None,
        help=(
            "Diagnostic: require this best-vs-second-best score gap before "
            "updating the adaptive CEM sampling center/sigma. Defaults to "
            "evaluate.py's 0.0 when omitted."
        ),
    )
    parser.add_argument(
        "--mjx-min-top-score-gap-override",
        action="append",
        default=[],
        metavar="MOTION:SEED:VALUE",
        help=(
            "Override --mjx-min-top-score-gap for one row, e.g. jump:0:0.03. "
            "Repeat for multiple rows."
        ),
    )
    parser.add_argument(
        "--mjx-min-score-improvement-override",
        action="append",
        default=[],
        metavar="MOTION:SEED:VALUE",
        help=(
            "Override --mjx-min-score-improvement for one row, e.g. jump:2:0.01. "
            "Repeat for multiple rows."
        ),
    )
    parser.add_argument(
        "--mjx-max-control-delta",
        type=float,
        default=None,
        help=(
            "Diagnostic: keep current controls when the selected MJX candidate "
            "exceeds this max absolute control delta."
        ),
    )
    parser.add_argument(
        "--mjx-max-control-delta-override",
        action="append",
        default=[],
        metavar="MOTION:SEED:VALUE",
        help=(
            "Override --mjx-max-control-delta for one row, e.g. walk:1:0.25. "
            "Repeat for multiple rows."
        ),
    )
    parser.add_argument(
        "--mjx-candidate-rank-diagnostics-top-k",
        type=int,
        default=0,
        help=(
            "Diagnostic: record per-iteration top-k MJX optimizer candidate "
            "indices and scores in MPC history. Defaults to 0/off."
        ),
    )
    parser.add_argument(
        "--mjx-candidate-rescore-diagnostics",
        action="store_true",
        help=(
            "Diagnostic: rescore the same sampled MJX optimizer candidates once "
            "per iteration and record score deltas. Defaults to off."
        ),
    )
    parser.add_argument(
        "--mjx-candidate-rescore-selection-top-k",
        type=int,
        default=0,
        help=(
            "Diagnostic: rescore the current top-k MJX optimizer candidates once "
            "per iteration and select using the average score. Defaults to 0/off."
        ),
    )
    parser.add_argument(
        "--mjx-candidate-score-component-diagnostics-top-k",
        type=int,
        default=0,
        help=(
            "Diagnostic: record score-component values for current, selected, "
            "and top-k MJX optimizer candidates. Defaults to 0/off."
        ),
    )
    parser.add_argument(
        "--mjx-score-only-rescore-diagnostics",
        action="store_true",
        help=(
            "Diagnostic: keep the primary MJX optimizer scorer unchanged, but "
            "shadow-rescore sampled candidates with a score-only scorer and "
            "record score deltas. Shadow scores are not used for selection."
        ),
    )
    parser.add_argument(
        "--mjx-score-only-output-rescore-diagnostics",
        action="store_true",
        help=(
            "Diagnostic: keep the primary MJX optimizer scorer unchanged, but "
            "shadow-rescore sampled candidates with a scorer that accumulates "
            "full metrics and returns only score. Shadow scores are not used "
            "for selection."
        ),
    )
    parser.add_argument(
        "--mjx-contact-force-first-row-diagnostics",
        action="store_true",
        help=(
            "Diagnostic: ask the MJX artifact trace to materialize first solver-row "
            "contact-force fields. This is off by default for formal speed runs."
        ),
    )
    parser.add_argument(
        "--mjx-contact-force-top-row-diagnostics",
        action="store_true",
        help=(
            "Diagnostic: ask the MJX artifact trace to materialize top "
            "MJX-Warp floor-contact row summaries. This is off by default "
            "for formal speed runs."
        ),
    )
    parser.add_argument(
        "--mjx-contact-force-mode",
        choices=("sum_rows", "first_row"),
        default="sum_rows",
        help=(
            "Diagnostic: choose which MJX contact-force signal enters scoring. "
            "The default sum_rows preserves the current acceptance surface; "
            "first_row is experimental and must be explicitly recorded."
        ),
    )
    parser.add_argument(
        "--mjx-contact-force-active-weight",
        type=float,
        default=0.0,
        help=(
            "Diagnostic MJX-only scorer weight for active foot contact-force "
            "magnitude. The term is normalized by 300N and defaults to 0/off."
        ),
    )
    parser.add_argument(
        "--mjx-contact-force-delta-weight",
        type=float,
        default=None,
        help=(
            "Diagnostic MJX-only override for the contact-force-delta scorer "
            "weight. Omit to preserve the method reward config."
        ),
    )
    parser.add_argument(
        "--mjx-contact-force-peak-excess-weight",
        type=float,
        default=0.0,
        help=(
            "Diagnostic MJX-only scorer weight for peak contact force above "
            "300N. The excess is normalized by 300N and defaults to 0/off."
        ),
    )
    parser.add_argument(
        "--mjx-contact-false-positive-weight",
        type=float,
        default=None,
        help=(
            "Diagnostic MJX-only override for the contact false-positive scorer "
            "weight. Omit to preserve the method reward config."
        ),
    )
    parser.add_argument(
        "--mjx-strip-live-mjx-data",
        action="store_true",
        help=(
            "Diagnostic: strip live MJX data between MPC windows so each window "
            "reconstructs MJX state from explicit qpos/qvel carry fields."
        ),
    )
    parser.add_argument(
        "--mjx-score-only-optimizer",
        action="store_true",
        help=(
            "Diagnostic: score optimizer candidates with compact score-only "
            "metrics. Formal acceptance must keep the default full metrics unless "
            "criteria promote this path."
        ),
    )
    parser.add_argument(
        "--mjx-mpc-samples",
        type=int,
        default=None,
        help=(
            "Diagnostic: MJX-only override for --mpc-samples. Replay validation "
            "keeps the Stage 0 sample count because it consumes the saved command."
        ),
    )
    parser.add_argument(
        "--mjx-mpc-root-pos-sigma",
        type=float,
        default=None,
        help="Diagnostic: MJX-only override for --mpc-root-pos-sigma.",
    )
    parser.add_argument(
        "--mjx-mpc-root-rot-sigma",
        type=float,
        default=None,
        help="Diagnostic: MJX-only override for --mpc-root-rot-sigma.",
    )
    parser.add_argument(
        "--mjx-mpc-joint-sigma",
        type=float,
        default=None,
        help="Diagnostic: MJX-only override for --mpc-joint-sigma.",
    )
    parser.add_argument(
        "--mjx-mpc-joint-sigma-override",
        action="append",
        default=[],
        metavar="MOTION:SEED:VALUE",
        help=(
            "Override MJX-only --mpc-joint-sigma for one row, e.g. walk:1:0.12. "
            "Repeat for multiple rows."
        ),
    )
    parser.add_argument(
        "--mjx-mpc-first-ctrl-noise-scale",
        type=float,
        default=None,
        help="Diagnostic: MJX-only override for --mpc-first-ctrl-noise-scale.",
    )
    parser.add_argument(
        "--mjx-mpc-last-ctrl-noise-scale",
        type=float,
        default=None,
        help="Diagnostic: MJX-only override for --mpc-last-ctrl-noise-scale.",
    )
    parser.add_argument(
        "--mjx-mpc-final-noise-scale",
        type=float,
        default=None,
        help="Diagnostic: MJX-only override for --mpc-final-noise-scale.",
    )
    parser.add_argument(
        "--mjx-mpc-sigma-decay",
        type=float,
        default=None,
        help="Diagnostic: MJX-only override for --mpc-sigma-decay.",
    )
    parser.add_argument(
        "--mjx-impl",
        choices=("jax", "warp"),
        default="jax",
        help="MJX implementation used for the formal MJX row.",
    )
    parser.add_argument(
        "--mjx-warp-naconmax",
        type=int,
        default=30000,
        help="Global MJX-Warp contact buffer passed to evaluate.py when using Warp.",
    )
    parser.add_argument(
        "--mjx-warp-naconmax-override",
        action="append",
        default=[],
        metavar="MOTION:SEED:VALUE",
        help=(
            "Override --mjx-warp-naconmax for one row, e.g. jump:1:24250. "
            "Repeat for multiple rows."
        ),
    )
    parser.add_argument(
        "--mjx-warp-njmax",
        type=int,
        default=256,
        help="Global MJX-Warp constraint buffer passed to evaluate.py when using Warp.",
    )
    parser.add_argument(
        "--mjx-model-iterations",
        type=int,
        default=None,
        help=(
            "Diagnostic MJX model opt.iterations override. Non-default values "
            "must be promoted in criteria before they are formal evidence."
        ),
    )
    parser.add_argument(
        "--mjx-model-iterations-override",
        action="append",
        default=[],
        metavar="MOTION:SEED:VALUE",
        help=(
            "Override --mjx-model-iterations for one row, e.g. walk:2:5. "
            "Repeat for multiple rows."
        ),
    )
    parser.add_argument(
        "--mjx-model-ls-iterations",
        type=int,
        default=None,
        help=(
            "Diagnostic MJX model opt.ls_iterations override. Non-default values "
            "must be promoted in criteria before they are formal evidence."
        ),
    )
    parser.add_argument(
        "--mjx-model-ls-iterations-override",
        action="append",
        default=[],
        metavar="MOTION:SEED:VALUE",
        help=(
            "Override --mjx-model-ls-iterations for one row, e.g. walk:2:5. "
            "Repeat for multiple rows."
        ),
    )
    parser.add_argument(
        "--only-motion",
        action="append",
        choices=MOTIONS,
        default=None,
        help="Run only the selected motion. Repeat to include multiple motions.",
    )
    parser.add_argument(
        "--only-seed",
        action="append",
        type=int,
        choices=SEEDS,
        default=None,
        help="Run only the selected seed. Repeat to include multiple seeds.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Run selected row shards without writing shared acceptance reports.",
    )
    args = parser.parse_args(argv)
    if args.mjx_model_iterations is not None and int(args.mjx_model_iterations) <= 0:
        parser.error("--mjx-model-iterations must be positive.")
    if (
        args.mjx_model_ls_iterations is not None
        and int(args.mjx_model_ls_iterations) <= 0
    ):
        parser.error("--mjx-model-ls-iterations must be positive.")
    if args.mjx_guided_candidate_period is not None:
        if int(args.mjx_guided_candidate_period) <= 0:
            parser.error("--mjx-guided-candidate-period must be positive.")
        if not bool(args.mjx_guided_candidate):
            parser.error(
                "--mjx-guided-candidate-period requires --mjx-guided-candidate."
            )
    if args.mjx_min_score_improvement is not None:
        min_score_improvement = float(args.mjx_min_score_improvement)
        if not math.isfinite(min_score_improvement) or min_score_improvement < 0.0:
            parser.error("--mjx-min-score-improvement must be non-negative.")
    for flag_name, value in (
        ("--mjx-min-top-score-gap", args.mjx_min_top_score_gap),
        (
            "--mjx-cem-update-min-top-score-gap",
            args.mjx_cem_update_min_top_score_gap,
        ),
    ):
        if value is None:
            continue
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0.0:
            parser.error(f"{flag_name} must be non-negative.")
    if args.mjx_max_control_delta is not None:
        max_control_delta = float(args.mjx_max_control_delta)
        if not math.isfinite(max_control_delta) or max_control_delta <= 0.0:
            parser.error("--mjx-max-control-delta must be positive.")
    contact_force_active_weight = float(args.mjx_contact_force_active_weight)
    if (
        not math.isfinite(contact_force_active_weight)
        or contact_force_active_weight < 0.0
    ):
        parser.error("--mjx-contact-force-active-weight must be non-negative.")
    if args.mjx_contact_force_delta_weight is not None:
        contact_force_delta_weight = float(args.mjx_contact_force_delta_weight)
        if (
            not math.isfinite(contact_force_delta_weight)
            or contact_force_delta_weight < 0.0
        ):
            parser.error("--mjx-contact-force-delta-weight must be non-negative.")
    contact_force_peak_excess_weight = float(
        args.mjx_contact_force_peak_excess_weight
    )
    if (
        not math.isfinite(contact_force_peak_excess_weight)
        or contact_force_peak_excess_weight < 0.0
    ):
        parser.error(
            "--mjx-contact-force-peak-excess-weight must be non-negative."
        )
    if args.mjx_contact_false_positive_weight is not None:
        contact_false_positive_weight = float(
            args.mjx_contact_false_positive_weight
        )
        if (
            not math.isfinite(contact_false_positive_weight)
            or contact_false_positive_weight < 0.0
        ):
            parser.error(
                "--mjx-contact-false-positive-weight must be non-negative."
            )
    if int(args.mjx_candidate_rank_diagnostics_top_k) < 0:
        parser.error("--mjx-candidate-rank-diagnostics-top-k must be non-negative.")
    if int(args.mjx_candidate_rescore_selection_top_k) < 0:
        parser.error("--mjx-candidate-rescore-selection-top-k must be non-negative.")
    if int(args.mjx_candidate_score_component_diagnostics_top_k) < 0:
        parser.error(
            "--mjx-candidate-score-component-diagnostics-top-k must be non-negative."
        )
    if args.mjx_mpc_samples is not None and int(args.mjx_mpc_samples) <= 0:
        parser.error("--mjx-mpc-samples must be positive.")
    for attr, flag in MJX_MPC_NUMERIC_OVERRIDE_FLAGS.items():
        value = getattr(args, attr)
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            parser.error(f"{flag} override must be finite.")
        if attr == "mjx_mpc_final_noise_scale":
            if numeric < 0.0:
                parser.error(f"{flag} override must be non-negative.")
        elif numeric <= 0.0:
            parser.error(f"{flag} override must be positive.")
    args.mjx_guided_candidate_period_overrides = _parse_row_positive_int_overrides(
        parser,
        args.mjx_guided_candidate_period_override,
        "--mjx-guided-candidate-period-override",
    )
    args.mjx_min_score_improvement_overrides = _parse_row_non_negative_float_overrides(
        parser,
        args.mjx_min_score_improvement_override,
        "--mjx-min-score-improvement-override",
    )
    args.mjx_min_top_score_gap_overrides = _parse_row_non_negative_float_overrides(
        parser,
        args.mjx_min_top_score_gap_override,
        "--mjx-min-top-score-gap-override",
    )
    args.mjx_max_control_delta_overrides = _parse_row_positive_float_overrides(
        parser,
        args.mjx_max_control_delta_override,
        "--mjx-max-control-delta-override",
    )
    args.mjx_mpc_joint_sigma_overrides = _parse_row_positive_float_overrides(
        parser,
        args.mjx_mpc_joint_sigma_override,
        "--mjx-mpc-joint-sigma-override",
    )
    args.mjx_warp_naconmax_overrides = _parse_mjx_warp_naconmax_overrides(
        parser,
        args.mjx_warp_naconmax_override,
    )
    args.mjx_model_iterations_overrides = _parse_row_positive_int_overrides(
        parser,
        args.mjx_model_iterations_override,
        "--mjx-model-iterations-override",
    )
    args.mjx_model_ls_iterations_overrides = _parse_row_positive_int_overrides(
        parser,
        args.mjx_model_ls_iterations_override,
        "--mjx-model-ls-iterations-override",
    )
    return args


def _parse_mjx_warp_naconmax_overrides(
    parser: argparse.ArgumentParser,
    raw_overrides: list[str],
) -> dict[tuple[str, int], int]:
    return _parse_row_positive_int_overrides(
        parser,
        raw_overrides,
        "--mjx-warp-naconmax-override",
    )


def _parse_row_positive_int_overrides(
    parser: argparse.ArgumentParser,
    raw_overrides: list[str],
    flag_name: str,
) -> dict[tuple[str, int], int]:
    overrides: dict[tuple[str, int], int] = {}
    for raw in raw_overrides:
        parts = str(raw).split(":")
        if len(parts) != 3:
            parser.error(f"{flag_name} must have form MOTION:SEED:VALUE.")
        motion, seed_text, value_text = parts
        if motion not in MOTIONS:
            parser.error(f"Unsupported row override motion: {motion!r}.")
        try:
            seed = int(seed_text)
            value = int(value_text)
        except ValueError:
            parser.error(f"{flag_name} seed and value must be integers.")
        if seed not in SEEDS:
            parser.error(f"Unsupported row override seed: {seed!r}.")
        if value <= 0:
            parser.error(f"{flag_name} value must be positive.")
        key = (motion, seed)
        if key in overrides:
            parser.error(f"Duplicate row override for {motion}/seed_{seed}.")
        overrides[key] = value
    return overrides


def _parse_row_non_negative_float_overrides(
    parser: argparse.ArgumentParser,
    raw_overrides: list[str],
    flag_name: str,
) -> dict[tuple[str, int], float]:
    overrides: dict[tuple[str, int], float] = {}
    for raw in raw_overrides:
        parts = str(raw).split(":")
        if len(parts) != 3:
            parser.error(f"{flag_name} must have form MOTION:SEED:VALUE.")
        motion, seed_text, value_text = parts
        if motion not in MOTIONS:
            parser.error(f"Unsupported row override motion: {motion!r}.")
        try:
            seed = int(seed_text)
            value = float(value_text)
        except ValueError:
            parser.error(f"{flag_name} seed must be integer and value must be float.")
        if seed not in SEEDS:
            parser.error(f"Unsupported row override seed: {seed!r}.")
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"{flag_name} value must be non-negative.")
        key = (motion, seed)
        if key in overrides:
            parser.error(f"Duplicate row override for {motion}/seed_{seed}.")
        overrides[key] = value
    return overrides


def _parse_row_positive_float_overrides(
    parser: argparse.ArgumentParser,
    raw_overrides: list[str],
    flag_name: str,
) -> dict[tuple[str, int], float]:
    overrides = _parse_row_non_negative_float_overrides(
        parser,
        raw_overrides,
        flag_name,
    )
    for (motion, seed), value in overrides.items():
        if value <= 0.0:
            parser.error(
                f"{flag_name} value must be positive for {motion}/seed_{seed}."
            )
    return overrides


def build_acceptance_plan(
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> list[PlannedAcceptanceRun]:
    output_root = args.output_dir.expanduser().resolve()
    rows = manifest.get("rows", [])
    matrix = _baseline_run_matrix(rows)
    _validate_formal_baseline_manifest(manifest, matrix)
    selected_motions = set(args.only_motion) if args.only_motion else set(MOTIONS)
    selected_seeds = set(args.only_seed) if args.only_seed else set(SEEDS)
    mjx_mpc_numeric_overrides = _mjx_mpc_numeric_overrides_from_args(args)
    plan: list[PlannedAcceptanceRun] = []
    for motion in MOTIONS:
        if motion not in selected_motions:
            continue
        for seed in SEEDS:
            if seed not in selected_seeds:
                continue
            row = matrix[(motion, seed)]
            output_dir = output_root / motion / f"seed_{seed}" / "mjx"
            replay_output_dir = output_root / motion / f"seed_{seed}" / "replay"
            row_key = (motion, seed)
            guided_period = args.mjx_guided_candidate_period_overrides.get(
                row_key,
                (
                    int(args.mjx_guided_candidate_period)
                    if args.mjx_guided_candidate_period is not None
                    else None
                ),
            )
            use_guided_candidate = bool(args.mjx_guided_candidate) or (
                row_key in args.mjx_guided_candidate_period_overrides
            )
            min_score_improvement = args.mjx_min_score_improvement_overrides.get(
                row_key,
                args.mjx_min_score_improvement,
            )
            min_top_score_gap = args.mjx_min_top_score_gap_overrides.get(
                row_key,
                args.mjx_min_top_score_gap,
            )
            max_control_delta = args.mjx_max_control_delta_overrides.get(
                row_key,
                args.mjx_max_control_delta,
            )
            row_mjx_mpc_numeric_overrides = dict(mjx_mpc_numeric_overrides)
            if row_key in args.mjx_mpc_joint_sigma_overrides:
                row_mjx_mpc_numeric_overrides["--mpc-joint-sigma"] = float(
                    args.mjx_mpc_joint_sigma_overrides[row_key]
                )
            mjx_argv = _mjx_argv_from_baseline_row(
                row,
                args.python_executable,
                output_dir,
                args.device,
                args.collision_profile,
                use_guided_candidate,
                args.mjx_impl,
                guided_period,
                int(
                    args.mjx_warp_naconmax_overrides.get(
                        row_key,
                        args.mjx_warp_naconmax,
                    )
                ),
                int(args.mjx_warp_njmax),
                _mjx_model_options_from_args(args, row_key=row_key),
                row_mjx_mpc_numeric_overrides,
                min_score_improvement,
                min_top_score_gap,
                args.mjx_cem_update_min_top_score_gap,
                max_control_delta,
                int(args.mjx_candidate_rank_diagnostics_top_k),
                bool(args.mjx_candidate_rescore_diagnostics),
                int(args.mjx_candidate_rescore_selection_top_k),
                int(args.mjx_candidate_score_component_diagnostics_top_k),
                bool(args.mjx_score_only_rescore_diagnostics),
                bool(args.mjx_score_only_output_rescore_diagnostics),
                bool(args.mjx_contact_force_first_row_diagnostics),
                bool(args.mjx_contact_force_top_row_diagnostics),
                str(args.mjx_contact_force_mode),
                float(args.mjx_contact_force_active_weight),
                args.mjx_contact_force_delta_weight,
                float(args.mjx_contact_force_peak_excess_weight),
                args.mjx_contact_false_positive_weight,
                bool(args.mjx_strip_live_mjx_data),
                bool(args.mjx_score_only_optimizer),
                args.mjx_mpc_samples,
            )
            replay_argv = _replay_argv_from_mjx(
                row,
                mjx_argv,
                output_dir,
                replay_output_dir,
            )
            plan.append(
                PlannedAcceptanceRun(
                    motion=motion,
                    seed=seed,
                    backend="mjx",
                    replay_backend="mujoco_warp",
                    output_dir=str(output_dir),
                    replay_output_dir=str(replay_output_dir),
                    mjx_argv=mjx_argv,
                    replay_argv=replay_argv,
                    mjx_command_text=shlex.join(mjx_argv),
                    replay_command_text=shlex.join(replay_argv),
                )
            )
    return plan


def _mjx_mpc_numeric_overrides_from_args(
    args: argparse.Namespace,
) -> dict[str, float]:
    return {
        flag: float(value)
        for attr, flag in MJX_MPC_NUMERIC_OVERRIDE_FLAGS.items()
        if (value := getattr(args, attr)) is not None
    }


def run_command(argv: list[str], *, cwd: Path) -> dict[str, Any]:
    command_start_time_ns = time.time_ns()
    start = time.perf_counter()
    result = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    wall_time_sec = time.perf_counter() - start
    row = {
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "status": "ok" if result.returncode == 0 else "failed",
        "command_wall_time_sec": float(wall_time_sec),
        "command_start_time_ns": int(command_start_time_ns),
    }
    metrics_path = _output_dir_from_argv(argv) / "metrics.json"
    if metrics_path.is_file():
        row.update(_row_from_metrics(metrics_path))
    return row


def write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ACCEPTANCE_REPORT_NAME
    _write_json_atomic(path, report)
    return path


def write_partial_report(
    output_dir: Path,
    *,
    plan: list[PlannedAcceptanceRun],
    mjx_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    run_status: str = "running",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ACCEPTANCE_PARTIAL_REPORT_NAME
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "run_status": str(run_status),
            "planned_runs": [asdict(item) for item in plan],
            "completed_mjx_rows": len(mjx_rows),
            "completed_replay_rows": len(replay_rows),
            "mjx_rows": mjx_rows,
            "replay_rows": replay_rows,
        },
    )
    return path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    baseline_manifest = args.baseline_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = json.loads(baseline_manifest.read_text())
    plan = build_acceptance_plan(args, manifest)
    environment_failures = validate_runtime_environment(args)
    if environment_failures:
        if args.skip_report:
            print(str(output_dir))
            return 1
        report = _environment_failure_report(
            baseline_manifest=baseline_manifest,
            manifest=manifest,
            args=args,
            failures=environment_failures,
        )
        report["planned_runs"] = [asdict(item) for item in plan]
        report_path = write_report(output_dir, report)
        print(str(report_path))
        return 1

    baseline_rows = list(manifest.get("rows", []))
    baseline_envelopes = _baseline_envelopes_from_manifest(manifest)
    if _baseline_artifact_preflight_failures(baseline_rows):
        if args.skip_report:
            print(str(output_dir))
            return 1
        report = _build_report(
            baseline_manifest=baseline_manifest,
            baseline_rows=baseline_rows,
            baseline_envelopes=baseline_envelopes,
            mjx_rows=[],
            replay_rows=[],
            min_speedup=float(args.min_speedup),
            target=str(args.target),
            min_realtime_factor=float(args.min_realtime_factor),
            required_gpu_name_fragment=str(args.required_gpu_name_fragment),
        )
        report["planned_runs"] = [asdict(item) for item in plan]
        report_path = write_report(output_dir, report)
        print(str(report_path))
        return 1

    mjx_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    existing_rows = (
        load_existing_acceptance_rows(output_dir)
        if args.reuse_existing_ok
        else {"mjx": [], "replay": []}
    )
    for planned in plan:
        if args.dry_run:
            mjx_row = {
                **asdict(planned),
                "status": "dry_run",
                "returncode": None,
                "metrics": {},
            }
            replay_row = {
                **asdict(planned),
                "status": "dry_run",
                "returncode": None,
                "metrics": {},
            }
        else:
            Path(planned.output_dir).mkdir(parents=True, exist_ok=True)
            Path(planned.replay_output_dir).mkdir(parents=True, exist_ok=True)
            mjx_row = load_existing_acceptance_row(
                planned,
                kind="mjx",
                existing_rows=existing_rows["mjx"],
            )
            if mjx_row is None:
                mjx_row = {
                    **asdict(planned),
                    **run_command(planned.mjx_argv, cwd=SPIDER_ROOT),
                }
            replay_row = load_existing_acceptance_row(
                planned,
                kind="replay",
                existing_rows=existing_rows["replay"],
            )
            if replay_row is None:
                replay_row = {
                    **asdict(planned),
                    **run_command(planned.replay_argv, cwd=SPIDER_ROOT),
                }
        attached_mjx_row = _attach_artifacts(mjx_row, planned.output_dir)
        attached_replay_row = _attach_artifacts(replay_row, planned.replay_output_dir)
        attached_mjx_row = _attach_single_row_speed_evidence(
            attached_mjx_row,
            baseline_rows,
            min_speedup=float(args.min_speedup),
            min_realtime_factor=float(args.min_realtime_factor),
        )
        mjx_rows.append(attached_mjx_row)
        replay_rows.append(attached_replay_row)
        if not args.dry_run:
            write_acceptance_row_sidecar(attached_mjx_row, planned.output_dir)
            write_acceptance_row_sidecar(attached_replay_row, planned.replay_output_dir)
        if not args.dry_run and not args.skip_report:
            write_partial_report(
                output_dir,
                plan=plan,
                mjx_rows=mjx_rows,
                replay_rows=replay_rows,
            )

    if args.skip_report:
        print(str(output_dir))
        failures = _skip_report_row_evidence_failures(
            mjx_rows,
            replay_rows,
            required_gpu_name_fragment=str(args.required_gpu_name_fragment),
        )
        return 0 if not failures else 1

    report = _build_report(
        baseline_manifest=baseline_manifest,
        baseline_rows=baseline_rows,
        baseline_envelopes=baseline_envelopes,
        mjx_rows=mjx_rows,
        replay_rows=replay_rows,
        min_speedup=float(args.min_speedup),
        target=str(args.target),
        min_realtime_factor=float(args.min_realtime_factor),
        required_gpu_name_fragment=str(args.required_gpu_name_fragment),
    )
    report["planned_runs"] = [asdict(item) for item in plan]
    report_path = write_report(output_dir, report)
    print(str(report_path))
    return 0 if bool(report["passed"]) else 1


def load_existing_acceptance_rows(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load reusable row candidates from reports and row sidecars."""

    candidates = [
        output_dir / ACCEPTANCE_REPORT_NAME,
        output_dir / ACCEPTANCE_PARTIAL_REPORT_NAME,
    ]
    existing = [path for path in candidates if path.is_file()]
    existing.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    rows_by_kind: dict[str, list[dict[str, Any]]] = {"mjx": [], "replay": []}
    for path in existing:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        paired_partial_keys = (
            _paired_acceptance_row_keys(payload)
            if path.name == ACCEPTANCE_PARTIAL_REPORT_NAME
            else None
        )
        for kind, field in (("mjx", "mjx_rows"), ("replay", "replay_rows")):
            rows = payload.get(field)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                key = _acceptance_row_key(row)
                if key is None:
                    continue
                if paired_partial_keys is not None and key not in paired_partial_keys:
                    continue
                row = dict(row)
                row["reuse_source_report"] = str(path)
                rows_by_kind[kind].append(row)
    for kind, row in _load_acceptance_row_sidecars(output_dir):
        rows_by_kind[kind].append(row)
    return rows_by_kind


def write_acceptance_row_sidecar(row: dict[str, Any], output_dir: str | Path) -> Path:
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / ACCEPTANCE_ROW_SIDECAR_NAME
    _write_json_atomic(path, row)
    return path


def _load_acceptance_row_sidecars(output_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(output_dir.rglob(ACCEPTANCE_ROW_SIDECAR_NAME)):
        kind = path.parent.name
        if kind not in {"mjx", "replay"}:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        key = _acceptance_row_key(payload)
        if key is None:
            continue
        row = dict(payload)
        row["reuse_source_sidecar"] = str(path)
        rows.append((kind, row))
    return rows


def _paired_acceptance_row_keys(payload: dict[str, Any]) -> set[tuple[str, int]]:
    return _acceptance_row_keys(payload.get("mjx_rows")) & _acceptance_row_keys(
        payload.get("replay_rows")
    )


def _acceptance_row_keys(rows: Any) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    if not isinstance(rows, list):
        return keys
    for row in rows:
        if isinstance(row, dict):
            key = _acceptance_row_key(row)
            if key is not None:
                keys.add(key)
    return keys


def _acceptance_row_key(row: dict[str, Any]) -> tuple[str, int] | None:
    motion = row.get("motion")
    seed = _safe_int(row.get("seed"))
    if not isinstance(motion, str) or seed is None:
        return None
    return motion, seed


def load_existing_acceptance_row(
    planned: PlannedAcceptanceRun,
    *,
    kind: str,
    existing_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return a verified reusable acceptance row for a planned run."""

    if kind not in {"mjx", "replay"}:
        raise ValueError(f"Unsupported acceptance row kind: {kind!r}")
    for row in existing_rows:
        if row.get("motion") != planned.motion:
            continue
        if _safe_int(row.get("seed")) != planned.seed:
            continue
        reusable = _verify_existing_acceptance_row(planned, kind=kind, row=row)
        if reusable is not None:
            return reusable
    return None


def _verify_existing_acceptance_row(
    planned: PlannedAcceptanceRun,
    *,
    kind: str,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    argv = planned.mjx_argv if kind == "mjx" else planned.replay_argv
    output_dir = Path(planned.output_dir if kind == "mjx" else planned.replay_output_dir)
    argv_field = "mjx_argv" if kind == "mjx" else "replay_argv"
    command_field = "mjx_command_text" if kind == "mjx" else "replay_command_text"
    expected_artifacts = (
        REQUIRED_ARTIFACT_FIELDS if kind == "mjx" else ("metrics_json", "rollout_npz")
    )

    if row.get("status") != "ok" or row.get("returncode") != 0:
        return None
    if row.get(argv_field) != argv:
        return None
    if row.get(command_field) != shlex.join(argv):
        return None
    row_output_field = "output_dir" if kind == "mjx" else "replay_output_dir"
    expected_output_dir = (
        planned.output_dir if kind == "mjx" else planned.replay_output_dir
    )
    if not _same_path(row.get(row_output_field), expected_output_dir):
        return None
    if not _artifacts_match_recorded_hashes(row, expected_artifacts):
        return None
    if _artifact_freshness_failures([row], artifact_fields=expected_artifacts):
        return None
    if _artifact_npz_schema_failures(
        [row],
        require_command=kind == "mjx",
        require_rollout_command_match=False,
    ):
        return None
    if kind == "mjx" and _mjx_dynamic_trace_evidence_failures([row]):
        return None

    metrics_path = output_dir / "metrics.json"
    try:
        metrics_payload = json.loads(metrics_path.read_text())
        parsed = _row_from_metrics(metrics_path)
    except (OSError, json.JSONDecodeError):
        return None
    if not _acceptance_metrics_provenance_matches(
        planned,
        kind=kind,
        payload=metrics_payload,
        parsed=parsed,
    ):
        return None

    reusable = dict(row)
    reusable.update(parsed)
    reusable["status"] = "ok"
    reusable["returncode"] = 0
    reusable["reused_existing"] = True
    reusable["reuse_kind"] = kind
    return reusable


def _artifacts_match_recorded_hashes(
    row: dict[str, Any],
    fields: tuple[str, ...],
) -> bool:
    artifacts = row.get("artifacts")
    hashes = row.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not isinstance(hashes, dict):
        return False
    for key in fields:
        path = artifacts.get(key)
        expected = hashes.get(key)
        if not isinstance(path, str) or not Path(path).expanduser().is_file():
            return False
        if not isinstance(expected, str) or not expected.strip():
            return False
        if _file_sha256(Path(path).expanduser()) != expected:
            return False
    return True


def _acceptance_metrics_provenance_matches(
    planned: PlannedAcceptanceRun,
    *,
    kind: str,
    payload: dict[str, Any],
    parsed: dict[str, Any],
) -> bool:
    argv = planned.mjx_argv if kind == "mjx" else planned.replay_argv
    parsed = dict(parsed)
    parsed["metrics_checkpoint"] = payload.get("checkpoint")
    parsed["metrics_max_steps"] = payload.get("max_steps")
    return _acceptance_metrics_provenance_matches_argv(
        argv,
        kind=kind,
        parsed=parsed,
    )


def _acceptance_metrics_provenance_matches_argv(
    argv: list[str],
    *,
    kind: str,
    parsed: dict[str, Any],
) -> bool:
    if parsed.get("metrics_method") != _argv_value(argv, "--method"):
        return False
    if not _same_path(parsed.get("metrics_motion"), _argv_value(argv, "--motion")):
        return False
    if parsed.get("metrics_device") != _argv_value(argv, "--device"):
        return False
    if not _same_path(parsed.get("metrics_checkpoint"), _argv_value(argv, "--checkpoint")):
        return False
    if _safe_int(parsed.get("metrics_max_steps")) != _safe_int(
        _argv_value(argv, "--max-steps")
    ):
        return False
    metrics = parsed.get("metrics")
    mpc = parsed.get("mpc")
    if not isinstance(metrics, dict) or not isinstance(mpc, dict):
        return False
    if _safe_int(parsed.get("num_steps", metrics.get("num_steps"))) != 800:
        return False
    if kind == "mjx":
        return _mjx_metrics_provenance_matches(argv, parsed, mpc)
    return _replay_metrics_provenance_matches(argv, parsed, mpc)


def _mjx_metrics_provenance_matches(
    argv: list[str],
    parsed: dict[str, Any],
    mpc: dict[str, Any],
) -> bool:
    if mpc.get("mpc_backend") != "mjx":
        return False
    expected_collision_profile = _argv_value(argv, "--collision-profile") or "wxy_parity"
    observed_collision_profile = mpc.get("collision_profile", parsed.get("collision_profile"))
    if observed_collision_profile is None:
        if expected_collision_profile != "wxy_parity":
            return False
    elif observed_collision_profile != expected_collision_profile:
        return False
    expected_impl = _argv_value(argv, "--mjx-impl") or "jax"
    if mpc.get("mjx_impl") != expected_impl:
        return False
    if expected_impl == "warp":
        if _safe_int(mpc.get("mjx_warp_naconmax")) != _safe_int(
            _argv_value(argv, "--mjx-warp-naconmax")
        ):
            return False
        if _safe_int(mpc.get("mjx_warp_njmax")) != _safe_int(
            _argv_value(argv, "--mjx-warp-njmax")
        ):
            return False
    expected_optimizer = _argv_value(argv, "--mpc-optimizer")
    if expected_optimizer is not None and mpc.get("mpc_optimizer") != expected_optimizer:
        return False
    if parsed.get("mpc_accepted") is not True:
        return False
    if _safe_int(parsed.get("accepted_windows")) != 40:
        return False
    if parsed.get("mpc_used_baseline_fallback") is not False:
        return False
    try:
        expected_guided_candidate = _argv_bool_optional(
            argv,
            "--mjx-guided-candidate",
        )
    except ValueError:
        return False
    if mpc.get("use_guided_candidate") is not expected_guided_candidate:
        return False
    expected_guided_period = _argv_value(argv, "--mjx-guided-candidate-period")
    if expected_guided_period is None:
        if mpc.get("guided_candidate_period") not in {None, 0}:
            return False
    elif _safe_int(mpc.get("guided_candidate_period")) != int(expected_guided_period):
        return False
    expected_guided_windows = _expected_guided_candidate_windows(
        use_guided_candidate=expected_guided_candidate,
        guided_candidate_period=expected_guided_period,
        accepted_windows=parsed.get("accepted_windows"),
    )
    if expected_guided_windows is None:
        return False
    if _safe_int(mpc.get("guided_candidate_windows")) != expected_guided_windows:
        return False
    try:
        expected_warm_start = _argv_bool_optional(argv, "--mpc-warm-start")
    except ValueError:
        return False
    if mpc.get("use_warm_start") is not expected_warm_start:
        return False
    for field, flag in (
        ("sample_count", "--mpc-samples"),
        ("optimizer_iterations", "--mpc-iterations"),
        ("planning_horizon_steps", "--mpc-planning-horizon-steps"),
        ("control_steps", "--mpc-control-steps"),
        ("knot_count", "--mpc-knot-count"),
    ):
        expected = _safe_int(_argv_value(argv, flag))
        if expected is None or _safe_int(mpc.get(field)) != expected:
            return False
    for field, flag in (
        ("elite_frac", "--mpc-elite-frac"),
        ("temperature", "--mpc-temperature"),
        ("root_pos_sigma", "--mpc-root-pos-sigma"),
        ("root_rot_sigma", "--mpc-root-rot-sigma"),
        ("joint_sigma", "--mpc-joint-sigma"),
        ("first_ctrl_noise_scale", "--mpc-first-ctrl-noise-scale"),
        ("last_ctrl_noise_scale", "--mpc-last-ctrl-noise-scale"),
        ("final_noise_scale", "--mpc-final-noise-scale"),
    ):
        expected = _argv_value(argv, flag)
        if expected is None or not _numeric_values_equivalent(
            mpc.get(field),
            float(expected),
        ):
            return False
    expected_sigma_decay = _argv_value(argv, "--mpc-sigma-decay")
    if expected_sigma_decay is not None and not _numeric_values_equivalent(
        mpc.get("sigma_decay"),
        float(expected_sigma_decay),
    ):
        return False
    expected_min_score_improvement = _argv_value(argv, "--mjx-min-score-improvement")
    try:
        expected_min_score_improvement_value = (
            1.0e-9
            if expected_min_score_improvement is None
            else float(expected_min_score_improvement)
        )
    except ValueError:
        return False
    if not _numeric_values_equivalent(
        mpc.get("mjx_min_score_improvement"),
        expected_min_score_improvement_value,
    ):
        return False
    expected_min_top_score_gap = _argv_value(argv, "--mjx-min-top-score-gap")
    try:
        expected_min_top_score_gap_value = (
            0.0
            if expected_min_top_score_gap is None
            else float(expected_min_top_score_gap)
        )
    except ValueError:
        return False
    if not _numeric_values_equivalent(
        mpc.get("mjx_min_top_score_gap"),
        expected_min_top_score_gap_value,
    ):
        return False
    expected_cem_update_min_top_score_gap = _argv_value(
        argv,
        "--mjx-cem-update-min-top-score-gap",
    )
    try:
        expected_cem_update_min_top_score_gap_value = (
            0.0
            if expected_cem_update_min_top_score_gap is None
            else float(expected_cem_update_min_top_score_gap)
        )
    except ValueError:
        return False
    if not _numeric_values_equivalent(
        mpc.get("mjx_cem_update_min_top_score_gap"),
        expected_cem_update_min_top_score_gap_value,
    ):
        return False
    expected_max_control_delta = _argv_value(argv, "--mjx-max-control-delta")
    try:
        expected_max_control_delta_value = (
            None
            if expected_max_control_delta is None
            else float(expected_max_control_delta)
        )
    except ValueError:
        return False
    observed_max_control_delta = mpc.get("mjx_max_control_delta")
    if expected_max_control_delta_value is None:
        if observed_max_control_delta is not None:
            return False
    elif not _numeric_values_equivalent(
        observed_max_control_delta,
        expected_max_control_delta_value,
    ):
        return False
    expected_candidate_rank_top_k = _argv_value(
        argv,
        "--mjx-candidate-rank-diagnostics-top-k",
    )
    try:
        expected_candidate_rank_top_k_value = (
            0
            if expected_candidate_rank_top_k is None
            else int(expected_candidate_rank_top_k)
        )
    except ValueError:
        return False
    if expected_candidate_rank_top_k_value < 0:
        return False
    observed_candidate_rank_top_k = mpc.get(
        "candidate_rank_diagnostics_top_k",
        mpc.get("mjx_candidate_rank_diagnostics_top_k"),
    )
    if expected_candidate_rank_top_k_value == 0:
        if observed_candidate_rank_top_k not in {0, None}:
            return False
    elif not _numeric_values_equivalent(
        observed_candidate_rank_top_k,
        expected_candidate_rank_top_k_value,
    ):
        return False
    expected_candidate_rescore_diagnostics = (
        "--mjx-candidate-rescore-diagnostics" in argv
    )
    observed_candidate_rescore_diagnostics = mpc.get(
        "mjx_candidate_rescore_diagnostics",
        mpc.get("candidate_rescore_diagnostics"),
    )
    observed_candidate_rescore_windows = mpc.get(
        "candidate_rescore_diagnostics_windows"
    )
    if expected_candidate_rescore_diagnostics:
        if observed_candidate_rescore_diagnostics is not True and not (
            isinstance(observed_candidate_rescore_windows, int)
            and observed_candidate_rescore_windows > 0
        ):
            return False
    elif observed_candidate_rescore_diagnostics not in {False, None}:
        return False
    elif (
        isinstance(observed_candidate_rescore_windows, int)
        and observed_candidate_rescore_windows > 0
    ):
        return False
    expected_candidate_rescore_selection_top_k = _argv_value(
        argv,
        "--mjx-candidate-rescore-selection-top-k",
    )
    try:
        expected_candidate_rescore_selection_top_k_value = (
            0
            if expected_candidate_rescore_selection_top_k is None
            else int(expected_candidate_rescore_selection_top_k)
        )
    except ValueError:
        return False
    if expected_candidate_rescore_selection_top_k_value < 0:
        return False
    observed_candidate_rescore_selection_top_k = mpc.get(
        "candidate_rescore_selection_top_k",
        mpc.get("mjx_candidate_rescore_selection_top_k"),
    )
    observed_candidate_rescore_selection_windows = mpc.get(
        "candidate_rescore_selection_windows"
    )
    if expected_candidate_rescore_selection_top_k_value == 0:
        if observed_candidate_rescore_selection_top_k not in {0, None}:
            return False
        if (
            isinstance(observed_candidate_rescore_selection_windows, int)
            and observed_candidate_rescore_selection_windows > 0
        ):
            return False
    elif not _numeric_values_equivalent(
        observed_candidate_rescore_selection_top_k,
        expected_candidate_rescore_selection_top_k_value,
    ):
        return False
    elif not (
        isinstance(observed_candidate_rescore_selection_windows, int)
        and observed_candidate_rescore_selection_windows > 0
    ):
        return False
    expected_score_component_top_k = _argv_value(
        argv,
        "--mjx-candidate-score-component-diagnostics-top-k",
    )
    try:
        expected_score_component_top_k_value = (
            0
            if expected_score_component_top_k is None
            else int(expected_score_component_top_k)
        )
    except ValueError:
        return False
    if expected_score_component_top_k_value < 0:
        return False
    observed_score_component_top_k = mpc.get(
        "candidate_score_component_diagnostics_top_k",
        mpc.get("mjx_candidate_score_component_diagnostics_top_k"),
    )
    if expected_score_component_top_k_value == 0:
        if observed_score_component_top_k not in {0, None}:
            return False
    elif not _numeric_values_equivalent(
        observed_score_component_top_k,
        expected_score_component_top_k_value,
    ):
        return False
    expected_first_row_diagnostics = "--mjx-contact-force-first-row-diagnostics" in argv
    observed_first_row_diagnostics = mpc.get(
        "contact_force_first_row_diagnostics"
    )
    if expected_first_row_diagnostics:
        if observed_first_row_diagnostics is not True:
            return False
    elif observed_first_row_diagnostics not in {False, None}:
        return False
    expected_top_row_diagnostics = "--mjx-contact-force-top-row-diagnostics" in argv
    observed_top_row_diagnostics = mpc.get("contact_force_top_row_diagnostics")
    if expected_top_row_diagnostics:
        if observed_top_row_diagnostics is not True:
            return False
    elif observed_top_row_diagnostics not in {False, None}:
        return False
    expected_contact_force_mode = _argv_value(argv, "--mjx-contact-force-mode")
    expected_contact_force_mode_value = (
        "sum_rows" if expected_contact_force_mode is None else expected_contact_force_mode
    )
    if expected_contact_force_mode_value not in {"sum_rows", "first_row"}:
        return False
    observed_contact_force_mode = mpc.get("contact_force_mode")
    if observed_contact_force_mode != expected_contact_force_mode_value:
        return False
    expected_contact_force_active_weight = _argv_value(
        argv,
        "--mjx-contact-force-active-weight",
    )
    try:
        expected_contact_force_active_weight_value = (
            0.0
            if expected_contact_force_active_weight is None
            else float(expected_contact_force_active_weight)
        )
    except ValueError:
        return False
    if expected_contact_force_active_weight_value < 0.0:
        return False
    observed_reward_weights = mpc.get("reward_weights") or {}
    if not isinstance(observed_reward_weights, Mapping):
        return False
    observed_contact_force_active_weight = observed_reward_weights.get(
        "contact_force_active",
        0.0,
    )
    if expected_contact_force_active_weight_value == 0.0:
        if not _numeric_values_equivalent(
            observed_contact_force_active_weight,
            0.0,
        ):
            return False
    elif not _numeric_values_equivalent(
        observed_contact_force_active_weight,
        expected_contact_force_active_weight_value,
    ):
        return False
    expected_contact_force_delta_weight = _argv_value(
        argv,
        "--mjx-contact-force-delta-weight",
    )
    if expected_contact_force_delta_weight is not None:
        try:
            expected_contact_force_delta_weight_value = float(
                expected_contact_force_delta_weight
            )
        except ValueError:
            return False
        if expected_contact_force_delta_weight_value < 0.0:
            return False
        observed_contact_force_delta_weight = observed_reward_weights.get(
            "contact_force_delta",
            0.0,
        )
        if not _numeric_values_equivalent(
            observed_contact_force_delta_weight,
            expected_contact_force_delta_weight_value,
        ):
            return False
    expected_contact_force_peak_excess_weight = _argv_value(
        argv,
        "--mjx-contact-force-peak-excess-weight",
    )
    try:
        expected_contact_force_peak_excess_weight_value = (
            0.0
            if expected_contact_force_peak_excess_weight is None
            else float(expected_contact_force_peak_excess_weight)
        )
    except ValueError:
        return False
    if expected_contact_force_peak_excess_weight_value < 0.0:
        return False
    observed_contact_force_peak_excess_weight = observed_reward_weights.get(
        "contact_force_peak_excess",
        0.0,
    )
    if expected_contact_force_peak_excess_weight_value == 0.0:
        if not _numeric_values_equivalent(
            observed_contact_force_peak_excess_weight,
            0.0,
        ):
            return False
    elif not _numeric_values_equivalent(
        observed_contact_force_peak_excess_weight,
        expected_contact_force_peak_excess_weight_value,
    ):
        return False
    expected_contact_false_positive_weight = _argv_value(
        argv,
        "--mjx-contact-false-positive-weight",
    )
    if expected_contact_false_positive_weight is not None:
        try:
            expected_contact_false_positive_weight_value = float(
                expected_contact_false_positive_weight
            )
        except ValueError:
            return False
        if expected_contact_false_positive_weight_value < 0.0:
            return False
        observed_contact_false_positive_weight = observed_reward_weights.get(
            "contact_false_positive",
            0.0,
        )
        if not _numeric_values_equivalent(
            observed_contact_false_positive_weight,
            expected_contact_false_positive_weight_value,
        ):
            return False
    expected_strip_live_mjx_data = "--mjx-strip-live-mjx-data" in argv
    observed_strip_live_mjx_data = mpc.get("strip_live_mjx_data_between_windows")
    if expected_strip_live_mjx_data:
        if observed_strip_live_mjx_data is not True:
            return False
    elif observed_strip_live_mjx_data not in {False, None}:
        return False
    expected_score_only_optimizer = "--mjx-score-only-optimizer" in argv
    observed_score_only_optimizer = mpc.get("score_only_optimizer")
    if expected_score_only_optimizer:
        if observed_score_only_optimizer is not True:
            return False
    elif observed_score_only_optimizer not in {False, None}:
        return False
    expected_score_only_rescore_diagnostics = (
        "--mjx-score-only-rescore-diagnostics" in argv
    )
    observed_score_only_rescore_diagnostics = mpc.get(
        "mjx_score_only_rescore_diagnostics",
        mpc.get("score_only_rescore_diagnostics"),
    )
    observed_score_only_rescore_windows = mpc.get(
        "score_only_rescore_diagnostics_windows"
    )
    if expected_score_only_rescore_diagnostics:
        if observed_score_only_rescore_diagnostics is not True and not (
            isinstance(observed_score_only_rescore_windows, int)
            and observed_score_only_rescore_windows > 0
        ):
            return False
    elif observed_score_only_rescore_diagnostics not in {False, None}:
        return False
    elif (
        isinstance(observed_score_only_rescore_windows, int)
        and observed_score_only_rescore_windows > 0
    ):
        return False
    expected_score_only_output_rescore_diagnostics = (
        "--mjx-score-only-output-rescore-diagnostics" in argv
    )
    observed_score_only_output_rescore_diagnostics = mpc.get(
        "mjx_score_only_output_rescore_diagnostics",
        mpc.get("score_only_output_rescore_diagnostics"),
    )
    observed_score_only_output_rescore_windows = mpc.get(
        "score_only_output_rescore_diagnostics_windows"
    )
    if expected_score_only_output_rescore_diagnostics:
        if observed_score_only_output_rescore_diagnostics is not True and not (
            isinstance(observed_score_only_output_rescore_windows, int)
            and observed_score_only_output_rescore_windows > 0
        ):
            return False
    elif observed_score_only_output_rescore_diagnostics not in {False, None}:
        return False
    elif (
        isinstance(observed_score_only_output_rescore_windows, int)
        and observed_score_only_output_rescore_windows > 0
    ):
        return False
    expected_model_options = _mjx_model_options_from_argv(argv)
    if expected_model_options is None:
        return False
    if not _mjx_model_options_match(
        mpc.get("mjx_model_options"),
        expected_model_options,
    ):
        return False
    for field in (
        "steady_state_wall_time_sec",
        "compile_init_wall_time_sec",
        "jit_warmup_wall_time_sec",
    ):
        if not _valid_timing(_row_value(parsed, field)):
            return False
    return True


def _expected_guided_candidate_windows(
    *,
    use_guided_candidate: bool,
    guided_candidate_period: Any,
    accepted_windows: Any,
) -> int | None:
    windows = _safe_int(accepted_windows)
    if windows is None or windows < 0:
        return None
    if not bool(use_guided_candidate):
        return 0
    if guided_candidate_period is None:
        return windows
    period = _safe_int(guided_candidate_period)
    if period is None or period <= 0:
        return None
    return (windows + period - 1) // period


def _replay_metrics_provenance_matches(
    argv: list[str],
    parsed: dict[str, Any],
    mpc: dict[str, Any],
) -> bool:
    if mpc.get("replay_mode") != "shared_execute_backend":
        return False
    saved = mpc.get("saved_command")
    expected_saved = _argv_value(argv, "--saved-command")
    if not _same_path(saved if isinstance(saved, str) else None, expected_saved):
        return False
    if not _is_existing_file(saved):
        return False
    saved_path = Path(str(saved)).expanduser()
    if mpc.get("saved_command_sha256") != _file_sha256(saved_path):
        return False
    expected_control = _safe_int(_argv_value(argv, "--replay-control-steps"))
    if _safe_int(mpc.get("control_steps")) != expected_control:
        return False
    if _safe_int(mpc.get("num_replay_steps")) != _safe_int(parsed.get("num_steps")):
        return False
    return True


def validate_runtime_environment(args: argparse.Namespace) -> tuple[str, ...]:
    """Return failures that make a real CUDA acceptance run non-formal."""

    if args.dry_run or not str(args.device).startswith("cuda"):
        return ()
    visible = tuple(
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    )
    failures: list[str] = []
    if len(visible) != 1:
        failures.append("single_gpu_visibility")
    if str(args.device) not in {"cuda", "cuda:0"}:
        failures.append("single_gpu_device")
    if len(visible) == 1 and _visible_gpu_has_compute_processes(visible[0]):
        failures.append("gpu_contention")
    return tuple(failures)


def _visible_gpu_has_compute_processes(visible_gpu: str) -> bool:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                str(visible_gpu),
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return True
    if result.returncode != 0:
        return True
    return any(line.strip() for line in result.stdout.splitlines())


def _environment_failure_report(
    *,
    baseline_manifest: Path,
    manifest: dict[str, Any],
    args: argparse.Namespace,
    failures: tuple[str, ...],
) -> dict[str, Any]:
    baseline_envelopes = _baseline_envelopes_from_manifest(manifest)
    return {
        "schema_version": 1,
        "backend": "mjx_canonical",
        "baseline_manifest": str(baseline_manifest),
        "contact_force_semantics": CONTACT_FORCE_SEMANTICS,
        "baseline_envelopes": baseline_envelopes,
        "classification": "invalid_benchmark",
        "target": str(args.target),
        "min_speedup": float(args.min_speedup),
        "min_realtime_factor": float(args.min_realtime_factor),
        "required_gpu_name_fragment": str(args.required_gpu_name_fragment),
        "environment_failures": tuple(failures),
        "motion_results": {},
        "replay_results": {},
        "speed_results": {},
        "realtime_results": {},
        "timing_summary": {},
        "contact_summary": {},
        "baseline_rows": list(manifest.get("rows", [])),
        "mjx_rows": [],
        "replay_rows": [],
        "passed": False,
    }


def _baseline_run_matrix(rows: Any) -> dict[tuple[str, int], dict[str, Any]]:
    expected = {(motion, seed) for motion in MOTIONS for seed in SEEDS}
    matrix: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: list[tuple[str, int]] = []
    extras: list[tuple[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        motion = str(row.get("motion_name"))
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError):
            extras.append((motion, row.get("seed")))
            continue
        key = (motion, seed)
        if key not in expected:
            extras.append(key)
            continue
        if key in matrix:
            duplicates.append(key)
        matrix[key] = row
    missing = sorted(expected - set(matrix))
    if missing or duplicates or extras:
        raise ValueError(
            "Baseline manifest run matrix must contain exactly one row for "
            f"{MOTIONS} x seeds {SEEDS}; "
            f"missing={missing}, duplicates={duplicates}, extras={extras}."
        )
    return matrix


def _validate_formal_baseline_manifest(
    manifest: dict[str, Any],
    matrix: dict[tuple[str, int], dict[str, Any]],
) -> None:
    failures: list[str] = []
    if manifest.get("schema_version") != 1:
        failures.append("schema_version")
    if manifest.get("baseline_name") != FORMAL_BASELINE_NAME:
        failures.append("baseline_name")
    if tuple(str(value) for value in manifest.get("motions", ())) != MOTIONS:
        failures.append("motions")
    try:
        seeds = tuple(int(value) for value in manifest.get("seeds", ()))
    except (TypeError, ValueError):
        seeds = ()
    if seeds != SEEDS:
        failures.append("seeds")
    if manifest.get("contact_force_semantics") != CONTACT_FORCE_SEMANTICS:
        failures.append("contact_force_semantics")
    failures.extend(_formal_manifest_provenance_failures(manifest, matrix))
    failures.extend(_formal_manifest_baseline_envelope_failures(manifest, matrix))

    for (motion, seed), row in matrix.items():
        failures.extend(_formal_stage0_row_failures(row, motion=motion, seed=seed))

    unique_failures = _unique(failures)
    if unique_failures:
        raise ValueError(
            "Baseline manifest is not a formal Stage 0 sweetpoint: "
            + ", ".join(unique_failures)
        )


def _formal_manifest_provenance_failures(
    manifest: dict[str, Any],
    matrix: dict[tuple[str, int], dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        failures.append("manifest_provenance")
    else:
        for field in ("worktree_path", "git_commit"):
            value = provenance.get(field)
            if not isinstance(value, str) or not value.strip():
                failures.append("manifest_provenance")
        status = provenance.get("git_status_short")
        if status is not None and not isinstance(status, str):
            failures.append("manifest_provenance")
    input_hashes = manifest.get("input_sha256")
    if not isinstance(input_hashes, dict):
        failures.append("manifest_input_sha256")
    else:
        for field in REQUIRED_INPUT_SHA256_FIELDS:
            if not _is_sha256_hex(input_hashes.get(field)):
                failures.append("manifest_input_sha256")
        if not _manifest_input_hashes_match(input_hashes, matrix):
            failures.append("manifest_input_sha256")
    return failures


def _formal_manifest_baseline_envelope_failures(
    manifest: dict[str, Any],
    matrix: dict[tuple[str, int], dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    baseline_envelopes = _baseline_envelopes_from_manifest(manifest)
    promoted_seeds = manifest.get("promoted_seeds")
    if set(baseline_envelopes) != set(MOTIONS):
        failures.append("baseline_envelopes")
    if not isinstance(promoted_seeds, dict):
        failures.append("promoted_seeds")
        promoted_seeds = {}

    for motion in MOTIONS:
        group = [matrix[(motion, seed)] for seed in SEEDS]
        gate = evaluate_baseline_group(motion, group)
        if not gate.passed:
            failures.append("baseline_envelopes")
            continue
        frozen = baseline_envelopes.get(motion)
        if frozen is None or not _envelopes_equivalent(frozen, gate.envelope):
            failures.append("baseline_envelopes")
        if _safe_int(promoted_seeds.get(motion)) != gate.promoted_seed:
            failures.append("promoted_seeds")
    return failures


def _baseline_envelopes_from_manifest(
    manifest: dict[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    raw = manifest.get("baseline_envelopes")
    if not isinstance(raw, dict):
        return {}
    baseline_envelopes: dict[str, dict[str, dict[str, float]]] = {}
    for motion in MOTIONS:
        envelope = _normalise_metric_envelope(raw.get(motion))
        if envelope is not None:
            baseline_envelopes[motion] = envelope
    return baseline_envelopes


def _normalise_metric_envelope(
    value: Any,
) -> dict[str, dict[str, float]] | None:
    if not isinstance(value, dict):
        return None
    envelope: dict[str, dict[str, float]] = {}
    for metric, stats in value.items():
        if not isinstance(metric, str) or not isinstance(stats, dict):
            return None
        envelope[metric] = {}
        for stat, stat_value in stats.items():
            if (
                not isinstance(stat, str)
                or isinstance(stat_value, bool)
                or not isinstance(stat_value, (int, float))
            ):
                return None
            stat_float = float(stat_value)
            if not math.isfinite(stat_float):
                return None
            envelope[metric][stat] = stat_float
    return envelope


def _envelopes_equivalent(
    observed: dict[str, dict[str, float]],
    expected: dict[str, dict[str, float]],
) -> bool:
    if set(observed) != set(expected):
        return False
    for metric, expected_stats in expected.items():
        observed_stats = observed.get(metric)
        if observed_stats is None or set(observed_stats) != set(expected_stats):
            return False
        for stat, expected_value in expected_stats.items():
            observed_value = observed_stats[stat]
            if not math.isclose(
                float(observed_value),
                float(expected_value),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                return False
    return True


def _manifest_input_hashes_match(
    input_hashes: dict[str, Any],
    matrix: dict[tuple[str, int], dict[str, Any]],
) -> bool:
    observed: dict[str, set[str]] = {
        field: set()
        for field in REQUIRED_INPUT_SHA256_FIELDS
    }
    for (motion, _seed), row in matrix.items():
        argv = row.get("argv")
        if not isinstance(argv, list):
            continue
        _record_file_hash(observed, f"{motion}_motion", _argv_value(argv, "--motion"))
        _record_file_hash(observed, "checkpoint", _argv_value(argv, "--checkpoint"))
        _record_file_hash(
            observed,
            "reward_weights",
            _argv_value(argv, "--mpc-reward-weights"),
        )
    for field in REQUIRED_INPUT_SHA256_FIELDS:
        if observed[field] != {input_hashes.get(field)}:
            return False
    return True


def _record_file_hash(
    observed: dict[str, set[str]],
    field: str,
    path: str | None,
) -> None:
    if field not in observed or not _is_existing_file(path):
        return
    observed[field].add(_file_sha256(Path(str(path)).expanduser()))


def _formal_stage0_row_failures(
    row: dict[str, Any],
    *,
    motion: str,
    seed: int,
) -> list[str]:
    failures: list[str] = []
    argv = row.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return ["argv"]

    for flag in (*FORMAL_STAGE0_ARG_VALUES, *FORMAL_STAGE0_DYNAMIC_ARG_FLAGS):
        if argv.count(flag) != 1:
            failures.append(flag)
    for flag, expected in FORMAL_STAGE0_ARG_VALUES.items():
        value = _argv_value(argv, flag)
        if value != expected:
            failures.append(flag)
    for flag in FORMAL_STAGE0_FLAGS:
        if argv.count(flag) != 1:
            failures.append(flag)
    for flag in FORMAL_STAGE0_FORBIDDEN_FLAGS:
        if flag in argv:
            failures.append(flag)

    if _argv_value(argv, "--seed") != str(seed):
        failures.append("--seed")
    if not _same_path(_argv_value(argv, "--motion"), row.get("motion")):
        failures.append("--motion")
    if not _same_path(_argv_value(argv, "--output-dir"), row.get("output_dir")):
        failures.append("--output-dir")

    failures.extend(
        _stage0_runner_provenance_failures(row, motion=motion, seed=seed)
    )
    failures.extend(_baseline_metrics_artifact_failures(row, argv=argv))

    motion_path = _argv_value(argv, "--motion")
    if not _is_existing_file(motion_path):
        failures.append("motion_file")
    checkpoint_path = _argv_value(argv, "--checkpoint")
    if not _is_existing_file(checkpoint_path):
        failures.append("checkpoint")
    reward_path = _argv_value(argv, "--mpc-reward-weights")
    if not _is_existing_file(reward_path):
        failures.append("--mpc-reward-weights")
    if row.get("motion_name") != motion:
        failures.append("motion_name")
    return failures


def _stage0_runner_provenance_failures(
    row: dict[str, Any],
    *,
    motion: str,
    seed: int,
) -> list[str]:
    output_dir = row.get("output_dir")
    if not isinstance(output_dir, str):
        return ["stage0_runner_provenance"]
    path = Path(output_dir).expanduser() / STAGE0_RUNNER_PROVENANCE_FILENAME
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return ["stage0_runner_provenance"]
    if not isinstance(payload, dict):
        return ["stage0_runner_provenance"]
    if payload.get("schema_version") != 1:
        return ["stage0_runner_provenance"]
    if payload.get("kind") != "g1_wbc_stage0_runner_provenance":
        return ["stage0_runner_provenance"]
    if payload.get("contact_force_semantics") != CONTACT_FORCE_SEMANTICS:
        return ["stage0_runner_provenance"]
    if payload.get("motion_name") != motion:
        return ["stage0_runner_provenance"]
    if _safe_int(payload.get("seed")) != seed:
        return ["stage0_runner_provenance"]
    if not _same_path(payload.get("motion"), row.get("motion")):
        return ["stage0_runner_provenance"]
    if not _same_path(payload.get("output_dir"), row.get("output_dir")):
        return ["stage0_runner_provenance"]
    return []


def _baseline_metrics_artifact_failures(
    row: dict[str, Any],
    *,
    argv: list[str],
) -> list[str]:
    artifacts = row.get("artifacts")
    metrics_path = artifacts.get("metrics_json") if isinstance(artifacts, dict) else None
    if not isinstance(metrics_path, str):
        return ["baseline_metrics_artifact"]
    path = Path(metrics_path).expanduser()
    if not path.is_file():
        return ["baseline_metrics_artifact"]
    try:
        payload = json.loads(path.read_text())
        parsed = _row_from_metrics(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, OverflowError):
        return ["baseline_metrics_artifact"]
    if not isinstance(payload, dict):
        return ["baseline_metrics_artifact"]
    if not _baseline_metrics_artifact_matches(row, argv=argv, payload=payload, parsed=parsed):
        return ["baseline_metrics_artifact"]
    return []


def _baseline_metrics_artifact_matches(
    row: dict[str, Any],
    *,
    argv: list[str],
    payload: dict[str, Any],
    parsed: dict[str, Any],
) -> bool:
    if not _metric_dicts_equivalent(parsed.get("metrics"), row.get("metrics")):
        return False
    if parsed.get("mpc_accepted") is not row.get("mpc_accepted"):
        return False
    if _safe_int(parsed.get("accepted_windows")) != _safe_int(row.get("accepted_windows")):
        return False
    if parsed.get("mpc_used_baseline_fallback") is not row.get(
        "mpc_used_baseline_fallback"
    ):
        return False
    if _safe_int(parsed.get("num_steps")) != _safe_int(row.get("num_steps")):
        return False
    if not _same_path(payload.get("motion"), _argv_value(argv, "--motion")):
        return False
    if not _same_path(payload.get("checkpoint"), _argv_value(argv, "--checkpoint")):
        return False
    if payload.get("motion_type") != _argv_value(argv, "--motion-type"):
        return False
    if payload.get("method") != _argv_value(argv, "--method"):
        return False
    if payload.get("device") != _argv_value(argv, "--device"):
        return False
    if _safe_int(payload.get("max_steps")) != _safe_int(_argv_value(argv, "--max-steps")):
        return False
    mpc = parsed.get("mpc")
    if not isinstance(mpc, dict):
        return False
    if mpc.get("mpc_backend") != _argv_value(argv, "--mpc-backend"):
        return False
    if mpc.get("mpc_optimizer") != _argv_value(argv, "--mpc-optimizer"):
        return False
    if not _same_path(
        mpc.get("reward_weight_source"),
        _argv_value(argv, "--mpc-reward-weights"),
    ):
        return False
    if mpc.get("config") != _expected_stage0_mpc_config(argv):
        return False
    row_timing = row.get("steady_state_wall_time_sec")
    if row_timing is not None and not _numeric_values_equivalent(
        parsed.get("steady_state_wall_time_sec"),
        row_timing,
    ):
        return False
    return True


def _metric_dicts_equivalent(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if set(left) != set(right):
        return False
    return all(_metric_values_equivalent(left[key], right[key]) for key in left)


def _metric_values_equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    return _numeric_values_equivalent(left, right)


def _numeric_values_equivalent(left: Any, right: Any) -> bool:
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, (int, float))
        or not isinstance(right, (int, float))
    ):
        return False
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _expected_stage0_mpc_config(argv: list[str]) -> dict[str, Any] | None:
    try:
        return {
            "mode": _required_argv_value(argv, "--method"),
            "num_samples": _argv_int(argv, "--mpc-samples"),
            "num_iterations": _argv_int(argv, "--mpc-iterations"),
            "planning_horizon_steps": _argv_int(
                argv,
                "--mpc-planning-horizon-steps",
            ),
            "control_steps": _argv_int(argv, "--mpc-control-steps"),
            "sampling_mode": _required_argv_value(argv, "--mpc-sampling-mode"),
            "knot_count": _argv_int(argv, "--mpc-knot-count"),
            "elite_frac": _argv_float(argv, "--mpc-elite-frac"),
            "temperature": _argv_float(argv, "--mpc-temperature"),
            "root_pos_sigma": _argv_float(argv, "--mpc-root-pos-sigma"),
            "root_rot_sigma": _argv_float(argv, "--mpc-root-rot-sigma"),
            "joint_sigma": _argv_float(argv, "--mpc-joint-sigma"),
            "min_root_pos_sigma": 0.002,
            "min_root_rot_sigma": 0.004,
            "min_joint_sigma": 0.008,
            "sigma_decay": _argv_float(argv, "--mpc-sigma-decay"),
            "smooth_passes": _argv_int(argv, "--mpc-smooth-passes"),
            "command_reg_weight": _argv_float(argv, "--mpc-command-reg-weight"),
            "command_smooth_weight": _argv_float(argv, "--mpc-command-smooth-weight"),
            "use_guided_candidate": _argv_bool_optional(
                argv,
                "--mpc-guided-candidate",
            ),
            "guided_root_pos_gain": _argv_float(argv, "--mpc-guided-root-pos-gain"),
            "guided_root_rot_gain": _argv_float(argv, "--mpc-guided-root-rot-gain"),
            "guided_joint_gain": _argv_float(argv, "--mpc-guided-joint-gain"),
            "guided_root_pos_clip": _argv_float(argv, "--mpc-guided-root-pos-clip"),
            "guided_root_rot_clip": _argv_float(argv, "--mpc-guided-root-rot-clip"),
            "guided_joint_clip": _argv_float(argv, "--mpc-guided-joint-clip"),
            "acceptance_gate": _argv_bool_optional(argv, "--mpc-acceptance-gate"),
            "seed": _argv_int(argv, "--seed"),
            "freeze_first_frame": True,
            "use_warm_start": _argv_bool_optional(argv, "--mpc-warm-start"),
            "warm_start_source": _required_argv_value(
                argv,
                "--mpc-warm-start-source",
            ),
            "warm_start_decay": _argv_float(argv, "--mpc-warm-start-decay"),
        }
    except (TypeError, ValueError):
        return None


def _required_argv_value(argv: list[str], flag: str) -> str:
    value = _argv_value(argv, flag)
    if value is None:
        raise ValueError(f"missing {flag}")
    return value


def _argv_int(argv: list[str], flag: str) -> int:
    return int(_required_argv_value(argv, flag))


def _argv_float(argv: list[str], flag: str) -> float:
    return float(_required_argv_value(argv, flag))


def _argv_bool_optional(argv: list[str], flag: str) -> bool:
    negative_flag = f"--no-{flag[2:]}"
    positive = flag in argv
    negative = negative_flag in argv
    if positive == negative:
        raise ValueError(f"expected exactly one of {flag} or {negative_flag}")
    return positive


def _mjx_model_options_from_args(
    args: argparse.Namespace,
    *,
    row_key: tuple[str, int] | None = None,
) -> dict[str, int]:
    options: dict[str, int] = {}
    iterations = args.mjx_model_iterations
    ls_iterations = args.mjx_model_ls_iterations
    if row_key is not None:
        iterations = args.mjx_model_iterations_overrides.get(row_key, iterations)
        ls_iterations = args.mjx_model_ls_iterations_overrides.get(
            row_key,
            ls_iterations,
        )
    if iterations is not None:
        options["iterations"] = int(iterations)
    if ls_iterations is not None:
        options["ls_iterations"] = int(ls_iterations)
    return options


def _mjx_model_options_from_argv(argv: list[str]) -> dict[str, int] | None:
    options: dict[str, int] = {}
    try:
        iterations = _argv_value(argv, "--mjx-model-iterations")
        if iterations is not None:
            options["iterations"] = int(iterations)
        ls_iterations = _argv_value(argv, "--mjx-model-ls-iterations")
        if ls_iterations is not None:
            options["ls_iterations"] = int(ls_iterations)
    except (TypeError, ValueError):
        return None
    return options


def _mjx_model_options_match(observed: Any, expected: dict[str, int]) -> bool:
    if observed is None:
        return expected == {}
    if not isinstance(observed, dict):
        return False
    if set(observed) != set(expected):
        return False
    for key, expected_value in expected.items():
        observed_value = observed.get(key)
        if isinstance(observed_value, bool) or not isinstance(
            observed_value,
            (int, float),
        ):
            return False
        if int(observed_value) != int(expected_value):
            return False
        if float(observed_value) != float(int(observed_value)):
            return False
    return True


def _mjx_argv_from_baseline_row(
    row: dict[str, Any],
    python_executable: str,
    output_dir: Path,
    device: str,
    collision_profile: str | None,
    use_guided_candidate: bool,
    mjx_impl: str,
    guided_candidate_period: int | None,
    mjx_warp_naconmax: int,
    mjx_warp_njmax: int,
    mjx_model_options: dict[str, int],
    mjx_mpc_numeric_overrides: dict[str, float],
    mjx_min_score_improvement: float | None,
    mjx_min_top_score_gap: float | None,
    mjx_cem_update_min_top_score_gap: float | None,
    mjx_max_control_delta: float | None,
    mjx_candidate_rank_diagnostics_top_k: int,
    mjx_candidate_rescore_diagnostics: bool,
    mjx_candidate_rescore_selection_top_k: int,
    mjx_candidate_score_component_diagnostics_top_k: int,
    mjx_score_only_rescore_diagnostics: bool,
    mjx_score_only_output_rescore_diagnostics: bool,
    mjx_contact_force_first_row_diagnostics: bool,
    mjx_contact_force_top_row_diagnostics: bool,
    mjx_contact_force_mode: str,
    mjx_contact_force_active_weight: float,
    mjx_contact_force_delta_weight: float | None,
    mjx_contact_force_peak_excess_weight: float,
    mjx_contact_false_positive_weight: float | None,
    mjx_strip_live_mjx_data: bool,
    mjx_score_only_optimizer: bool,
    mjx_mpc_samples: int | None,
) -> list[str]:
    argv = list(row["argv"])
    argv[0] = str(python_executable)
    argv = _set_arg(argv, "--mpc-backend", "mjx")
    argv = _set_arg(argv, "--mpc-optimizer", "generic")
    argv = _set_arg(argv, "--output-dir", str(output_dir))
    argv = _set_arg(argv, "--device", str(device))
    if collision_profile is not None:
        argv = _set_arg(argv, "--collision-profile", str(collision_profile))
    for flag in LEGACY_ONLY_MJX_DROP_ARG_VALUES:
        argv = _drop_arg_with_value(argv, flag)
    for flag in LEGACY_ONLY_MJX_DROP_FLAGS:
        argv = _drop_flag(argv, flag)
    for flag, value in MJX_GENERIC_ARG_VALUES.items():
        argv = _set_arg(argv, flag, value)
    for flag, value in mjx_mpc_numeric_overrides.items():
        argv = _set_arg(argv, flag, str(float(value)))
    if mjx_mpc_samples is not None:
        argv = _set_arg(argv, "--mpc-samples", str(int(mjx_mpc_samples)))
    if "--mjx-enable-scan" not in argv:
        argv.append("--mjx-enable-scan")
    argv = _set_arg(argv, "--mjx-impl", str(mjx_impl))
    argv = _set_arg(argv, "--mjx-warp-naconmax", str(int(mjx_warp_naconmax)))
    argv = _set_arg(argv, "--mjx-warp-njmax", str(int(mjx_warp_njmax)))
    for flag in ("--mjx-model-iterations", "--mjx-model-ls-iterations"):
        argv = _drop_arg_with_value(argv, flag)
    if "iterations" in mjx_model_options:
        argv = _set_arg(
            argv,
            "--mjx-model-iterations",
            str(int(mjx_model_options["iterations"])),
        )
    if "ls_iterations" in mjx_model_options:
        argv = _set_arg(
            argv,
            "--mjx-model-ls-iterations",
            str(int(mjx_model_options["ls_iterations"])),
        )
    argv = _drop_arg_with_value(argv, "--mjx-min-score-improvement")
    if mjx_min_score_improvement is not None:
        argv = _set_arg(
            argv,
            "--mjx-min-score-improvement",
            str(float(mjx_min_score_improvement)),
        )
    argv = _drop_arg_with_value(argv, "--mjx-min-top-score-gap")
    if mjx_min_top_score_gap is not None:
        argv = _set_arg(
            argv,
            "--mjx-min-top-score-gap",
            str(float(mjx_min_top_score_gap)),
        )
    argv = _drop_arg_with_value(argv, "--mjx-cem-update-min-top-score-gap")
    if mjx_cem_update_min_top_score_gap is not None:
        argv = _set_arg(
            argv,
            "--mjx-cem-update-min-top-score-gap",
            str(float(mjx_cem_update_min_top_score_gap)),
        )
    argv = _drop_arg_with_value(argv, "--mjx-max-control-delta")
    if mjx_max_control_delta is not None:
        argv = _set_arg(
            argv,
            "--mjx-max-control-delta",
            str(float(mjx_max_control_delta)),
        )
    argv = _drop_arg_with_value(argv, "--mjx-candidate-rank-diagnostics-top-k")
    if int(mjx_candidate_rank_diagnostics_top_k) > 0:
        argv = _set_arg(
            argv,
            "--mjx-candidate-rank-diagnostics-top-k",
            str(int(mjx_candidate_rank_diagnostics_top_k)),
        )
    argv = _drop_flag(argv, "--mjx-candidate-rescore-diagnostics")
    if bool(mjx_candidate_rescore_diagnostics):
        argv.append("--mjx-candidate-rescore-diagnostics")
    argv = _drop_arg_with_value(argv, "--mjx-candidate-rescore-selection-top-k")
    if int(mjx_candidate_rescore_selection_top_k) > 0:
        argv = _set_arg(
            argv,
            "--mjx-candidate-rescore-selection-top-k",
            str(int(mjx_candidate_rescore_selection_top_k)),
        )
    argv = _drop_arg_with_value(
        argv, "--mjx-candidate-score-component-diagnostics-top-k"
    )
    if int(mjx_candidate_score_component_diagnostics_top_k) > 0:
        argv = _set_arg(
            argv,
            "--mjx-candidate-score-component-diagnostics-top-k",
            str(int(mjx_candidate_score_component_diagnostics_top_k)),
        )
    argv = _drop_flag(argv, "--mjx-score-only-rescore-diagnostics")
    if bool(mjx_score_only_rescore_diagnostics):
        argv.append("--mjx-score-only-rescore-diagnostics")
    argv = _drop_flag(argv, "--mjx-score-only-output-rescore-diagnostics")
    if bool(mjx_score_only_output_rescore_diagnostics):
        argv.append("--mjx-score-only-output-rescore-diagnostics")
    argv = _drop_flag(argv, "--mjx-contact-force-first-row-diagnostics")
    if bool(mjx_contact_force_first_row_diagnostics):
        argv.append("--mjx-contact-force-first-row-diagnostics")
    argv = _drop_flag(argv, "--mjx-contact-force-top-row-diagnostics")
    if bool(mjx_contact_force_top_row_diagnostics):
        argv.append("--mjx-contact-force-top-row-diagnostics")
    argv = _drop_arg_with_value(argv, "--mjx-contact-force-mode")
    if str(mjx_contact_force_mode) != "sum_rows":
        argv = _set_arg(
            argv,
            "--mjx-contact-force-mode",
            str(mjx_contact_force_mode),
        )
    argv = _drop_arg_with_value(argv, "--mjx-contact-force-active-weight")
    if float(mjx_contact_force_active_weight) > 0.0:
        argv = _set_arg(
            argv,
            "--mjx-contact-force-active-weight",
            str(float(mjx_contact_force_active_weight)),
        )
    argv = _drop_arg_with_value(argv, "--mjx-contact-force-delta-weight")
    if mjx_contact_force_delta_weight is not None:
        argv = _set_arg(
            argv,
            "--mjx-contact-force-delta-weight",
            str(float(mjx_contact_force_delta_weight)),
        )
    argv = _drop_arg_with_value(argv, "--mjx-contact-force-peak-excess-weight")
    if float(mjx_contact_force_peak_excess_weight) > 0.0:
        argv = _set_arg(
            argv,
            "--mjx-contact-force-peak-excess-weight",
            str(float(mjx_contact_force_peak_excess_weight)),
        )
    argv = _drop_arg_with_value(argv, "--mjx-contact-false-positive-weight")
    if mjx_contact_false_positive_weight is not None:
        argv = _set_arg(
            argv,
            "--mjx-contact-false-positive-weight",
            str(float(mjx_contact_false_positive_weight)),
        )
    argv = _drop_flag(argv, "--mjx-strip-live-mjx-data")
    if bool(mjx_strip_live_mjx_data):
        argv.append("--mjx-strip-live-mjx-data")
    argv = _drop_flag(argv, "--mjx-score-only-optimizer")
    if bool(mjx_score_only_optimizer):
        argv.append("--mjx-score-only-optimizer")
    if "--save-rollout" not in argv:
        argv.append("--save-rollout")
    for flag in ("--mjx-guided-candidate", "--no-mjx-guided-candidate"):
        argv = _drop_flag(argv, flag)
    argv.append(
        "--mjx-guided-candidate"
        if use_guided_candidate
        else "--no-mjx-guided-candidate"
    )
    argv = _drop_arg_with_value(argv, "--mjx-guided-candidate-period")
    if guided_candidate_period is not None:
        argv = _set_arg(
            argv,
            "--mjx-guided-candidate-period",
            str(int(guided_candidate_period)),
        )
    return argv


def _replay_argv_from_mjx(
    baseline_row: dict[str, Any],
    mjx_argv: list[str],
    mjx_output_dir: Path,
    replay_output_dir: Path,
) -> list[str]:
    argv = list(mjx_argv)
    argv = _set_arg(argv, "--method", "replay_command")
    argv = _set_arg(argv, "--mpc-backend", "mujoco_warp")
    argv = _set_arg(argv, "--output-dir", str(replay_output_dir))
    argv = _drop_arg_with_value(argv, "--mpc-reward-weights")
    argv = _drop_arg_with_value(argv, "--mjx-impl")
    argv = _drop_arg_with_value(argv, "--mjx-warp-naconmax")
    argv = _drop_arg_with_value(argv, "--mjx-warp-njmax")
    argv = _drop_arg_with_value(argv, "--mjx-model-iterations")
    argv = _drop_arg_with_value(argv, "--mjx-model-ls-iterations")
    argv = _drop_arg_with_value(argv, "--mjx-min-score-improvement")
    argv = _drop_arg_with_value(argv, "--mjx-min-top-score-gap")
    argv = _drop_arg_with_value(argv, "--mjx-cem-update-min-top-score-gap")
    argv = _drop_arg_with_value(argv, "--mjx-max-control-delta")
    argv = _drop_arg_with_value(argv, "--mjx-candidate-rank-diagnostics-top-k")
    argv = _drop_flag(argv, "--mjx-candidate-rescore-diagnostics")
    argv = _drop_arg_with_value(argv, "--mjx-candidate-rescore-selection-top-k")
    argv = _drop_arg_with_value(
        argv, "--mjx-candidate-score-component-diagnostics-top-k"
    )
    argv = _drop_flag(argv, "--mjx-score-only-rescore-diagnostics")
    argv = _drop_flag(argv, "--mjx-score-only-output-rescore-diagnostics")
    argv = _drop_flag(argv, "--mjx-contact-force-first-row-diagnostics")
    argv = _drop_flag(argv, "--mjx-contact-force-top-row-diagnostics")
    argv = _drop_arg_with_value(argv, "--mjx-contact-force-mode")
    argv = _drop_arg_with_value(argv, "--mjx-contact-force-active-weight")
    argv = _drop_arg_with_value(argv, "--mjx-contact-force-delta-weight")
    argv = _drop_arg_with_value(argv, "--mjx-contact-force-peak-excess-weight")
    argv = _drop_arg_with_value(argv, "--mjx-contact-false-positive-weight")
    argv = _drop_flag(argv, "--mjx-strip-live-mjx-data")
    argv = _drop_flag(argv, "--mjx-score-only-optimizer")
    for flag in MJX_MPC_NUMERIC_OVERRIDE_FLAGS.values():
        if flag in FORMAL_STAGE0_ARG_VALUES:
            argv = _set_arg(argv, flag, FORMAL_STAGE0_ARG_VALUES[flag])
        elif flag in MJX_GENERIC_ARG_VALUES:
            argv = _set_arg(argv, flag, MJX_GENERIC_ARG_VALUES[flag])
        else:
            argv = _drop_arg_with_value(argv, flag)
    argv = _set_arg(argv, "--mpc-samples", FORMAL_STAGE0_ARG_VALUES["--mpc-samples"])
    argv = _drop_arg_with_value(argv, "--mpc-sigma-decay")
    argv = _drop_flag(argv, "--mjx-enable-scan")
    argv = _drop_flag(argv, "--mjx-guided-candidate")
    argv = _drop_flag(argv, "--no-mjx-guided-candidate")
    argv = _drop_arg_with_value(argv, "--mjx-guided-candidate-period")
    argv.extend(["--saved-command", str(mjx_output_dir / "mpc_command.npz")])
    argv.extend(["--replay-control-steps", "20"])
    argv.extend(["--replay-task-mode", "g1_wbc_joint_global"])
    argv = _set_arg(argv, "--seed", str(int(baseline_row["seed"])))
    return argv


def _build_report(
    *,
    baseline_manifest: Path,
    baseline_rows: list[dict[str, Any]],
    baseline_envelopes: dict[str, dict[str, dict[str, float]]],
    mjx_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    min_speedup: float,
    target: str,
    min_realtime_factor: float,
    required_gpu_name_fragment: str,
) -> dict[str, Any]:
    motion_results = {}
    replay_results = {}
    speed_results = {}
    realtime_results = {}
    passed = True
    for motion in MOTIONS:
        baseline_group = [row for row in baseline_rows if row.get("motion_name") == motion]
        mjx_group = [row for row in mjx_rows if row.get("motion") == motion]
        replay_group = [row for row in replay_rows if row.get("motion") == motion]
        baseline_gate = evaluate_baseline_group(motion, baseline_group)
        baseline_envelope = baseline_envelopes[motion]
        baseline_failures = _unique(
            (
                *baseline_gate.failures,
                *_baseline_artifact_failures(baseline_group),
                *_baseline_runtime_evidence_failures(
                    baseline_group,
                    required_gpu_name_fragment=required_gpu_name_fragment,
                ),
            )
        )
        mjx_gate = evaluate_mjx_group(
            motion,
            mjx_group,
            baseline_envelope,
            MjxQualityPolicy.for_motion(motion),
        )
        mjx_failures = _unique(
            (
                *mjx_gate.failures,
                *_acceptance_row_metrics_provenance_failures(
                    mjx_group,
                    kind="mjx",
                ),
                *_mjx_timing_evidence_failures(mjx_group),
                *_mjx_runtime_evidence_failures(
                    mjx_group,
                    required_gpu_name_fragment=required_gpu_name_fragment,
                ),
                *_mjx_contact_evidence_failures(mjx_group),
                *_mjx_dynamic_trace_evidence_failures(mjx_group),
                *_artifact_freshness_failures(
                    mjx_group,
                    artifact_fields=REQUIRED_ARTIFACT_FIELDS,
                ),
                *_artifact_hash_failures(
                    mjx_group,
                    artifact_fields=REQUIRED_ARTIFACT_FIELDS,
                ),
                *_artifact_npz_schema_failures(
                    mjx_group,
                    require_command=True,
                    require_rollout_command_match=False,
                ),
            )
        )
        replay_failures = _row_evidence_failures(
            replay_group,
            timing_failure="replay_steady_state_wall_time",
            timing_fields=("command_wall_time_sec", "steady_state_wall_time_sec"),
            artifact_fields=("metrics_json", "rollout_npz"),
            require_mpc_fields=False,
        )
        replay_failures = _unique(
            (
                *replay_failures,
                *_replay_provenance_failures(replay_group, mjx_group),
                *_runtime_evidence_failures(
                    replay_group,
                    prefix="replay",
                    required_gpu_name_fragment=required_gpu_name_fragment,
                ),
                *_artifact_freshness_failures(
                    replay_group,
                    artifact_fields=("metrics_json", "rollout_npz"),
                ),
                *_artifact_hash_failures(
                    replay_group,
                    artifact_fields=("metrics_json", "rollout_npz"),
                ),
                *_artifact_npz_schema_failures(
                    replay_group,
                    require_command=False,
                ),
                *_replay_quality_failures(
                    motion,
                    replay_group,
                    baseline_envelope,
                ),
            )
        )
        replay_passed = not replay_failures
        speed_gate = _speed_gate_for_motion(
            baseline_group,
            mjx_group,
            min_speedup=min_speedup,
        )
        motion_results[motion] = {
            "baseline_passed": not baseline_failures,
            "baseline_failures": baseline_failures,
            "mjx_passed": not mjx_failures,
            "mjx_failures": mjx_failures,
        }
        replay_results[motion] = {
            "passed": replay_passed,
            "failures": replay_failures,
        }
        speed_results[motion] = {
            "passed": speed_gate.passed,
            "speedup": speed_gate.speedup,
            "worst_speedup": speed_gate.worst_speedup,
            "failures": speed_gate.failures,
        }
        realtime_gate = _realtime_gate_for_motion(
            mjx_group,
            min_realtime_factor=min_realtime_factor,
            require_explicit_duration=target == TARGET_4090_REALTIME,
        )
        realtime_results[motion] = realtime_gate
        target_speed_passed = (
            speed_gate.passed and bool(realtime_gate["passed"])
            if target == TARGET_4090_REALTIME
            else speed_gate.passed
        )
        passed = (
            passed
            and not baseline_failures
            and not mjx_failures
            and replay_passed
            and target_speed_passed
        )
    classification = _classify_report(
        motion_results=motion_results,
        replay_results=replay_results,
        speed_results=speed_results,
        realtime_results=realtime_results,
        target=target,
        passed=passed,
    )
    return {
        "schema_version": 1,
        "backend": "mjx_canonical",
        "baseline_manifest": str(baseline_manifest),
        "contact_force_semantics": CONTACT_FORCE_SEMANTICS,
        "baseline_envelopes": baseline_envelopes,
        "classification": classification,
        "target": target,
        "min_speedup": float(min_speedup),
        "min_realtime_factor": float(min_realtime_factor),
        "required_gpu_name_fragment": required_gpu_name_fragment,
        "motion_results": motion_results,
        "replay_results": replay_results,
        "speed_results": speed_results,
        "realtime_results": realtime_results,
        "timing_summary": _timing_summary(
            baseline_rows=baseline_rows,
            mjx_rows=mjx_rows,
            replay_rows=replay_rows,
        ),
        "contact_summary": _contact_summary(
            baseline_rows=baseline_rows,
            mjx_rows=mjx_rows,
            replay_rows=replay_rows,
        ),
        "baseline_rows": baseline_rows,
        "mjx_rows": mjx_rows,
        "replay_rows": replay_rows,
        "passed": passed,
    }


def _timing_summary(
    *,
    baseline_rows: list[dict[str, Any]],
    mjx_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for motion in MOTIONS:
        baseline_group = [row for row in baseline_rows if row.get("motion_name") == motion]
        mjx_group = [row for row in mjx_rows if row.get("motion") == motion]
        replay_group = [row for row in replay_rows if row.get("motion") == motion]
        summary[motion] = {
            "baseline": {
                "steady_state_wall_time_sec": _timing_stats(
                    _timing_values(baseline_group, "steady_state_wall_time_sec")
                ),
            },
            "mjx": {
                "steady_state_wall_time_sec": _timing_stats(
                    _timing_values(mjx_group, "steady_state_wall_time_sec")
                ),
                "compile_init_wall_time_sec": _timing_stats(
                    _timing_values(mjx_group, "compile_init_wall_time_sec")
                ),
                "jit_warmup_wall_time_sec": _timing_stats(
                    _timing_values(mjx_group, "jit_warmup_wall_time_sec")
                ),
                "per_window_steady_state_wall_time_sec": _timing_stats(
                    _per_window_timing_values(mjx_group)
                ),
                "num_windows": _timing_stats(_window_count_values(mjx_group)),
                "runtime_visible_devices": [
                    [str(value) for value in row.get("runtime_visible_devices", ())]
                    for row in mjx_group
                ],
                "runtime_gpu_names": [
                    row.get("runtime_gpu_name")
                    if isinstance(row.get("runtime_gpu_name"), str)
                    else None
                    for row in mjx_group
                ],
            },
            "replay": {
                "command_wall_time_sec": _timing_stats(
                    _timing_values(replay_group, "command_wall_time_sec")
                ),
                "steady_state_wall_time_sec": _timing_stats(
                    _timing_values(replay_group, "steady_state_wall_time_sec")
                ),
            },
        }
    return summary


def _contact_summary(
    *,
    baseline_rows: list[dict[str, Any]],
    mjx_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for motion in MOTIONS:
        mjx_group = [row for row in mjx_rows if row.get("motion") == motion]
        baseline_group = [
            row for row in baseline_rows if row.get("motion_name") == motion
        ]
        replay_group = [row for row in replay_rows if row.get("motion") == motion]
        summary[motion] = {
            "baseline": {
                "force": _force_metrics_summary(baseline_group),
            },
            "mjx": {
                "max_contact_points": _timing_stats(
                    _contact_count_values(mjx_group, "max_contact_points")
                ),
                "max_geom_pairs": _timing_stats(
                    _contact_count_values(mjx_group, "max_geom_pairs")
                ),
                "contact_pair_count": _timing_stats(
                    _contact_count_values(mjx_group, "contact_pair_count")
                ),
                "active_contact_count": _timing_stats(
                    _contact_count_values(mjx_group, "active_contact_count")
                ),
                "saturation_flags": {
                    field: [_contact_flag_value(row, field) for row in mjx_group]
                    for field in MJX_CONTACT_SATURATION_FIELDS
                },
                "force": _force_metrics_summary(mjx_group),
            },
            "replay": {
                "force": _force_metrics_summary(replay_group),
            },
            "force_semantics": _force_semantics_summary(
                baseline_rows=baseline_group,
                mjx_rows=mjx_group,
                replay_rows=replay_group,
            ),
        }
    return summary


def _force_metrics_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "contact_force_active_mean": _timing_stats(
            _metric_float_values(rows, "contact_force_active_mean")
        ),
        "contact_force_peak": _timing_stats(
            _metric_float_values(rows, "contact_force_peak")
        ),
        "contact_force_first_row_active_mean": _timing_stats(
            _metric_float_values(rows, "contact_force_first_row_active_mean")
        ),
        "contact_force_first_row_peak": _timing_stats(
            _metric_float_values(rows, "contact_force_first_row_peak")
        ),
        "contact_force_sum_to_first_row_active_ratio": _timing_stats(
            _metric_float_values(rows, "contact_force_sum_to_first_row_active_ratio")
        ),
        "contact_force_sum_to_first_row_peak_ratio": _timing_stats(
            _metric_float_values(rows, "contact_force_sum_to_first_row_peak_ratio")
        ),
    }


def _force_semantics_summary(
    *,
    baseline_rows: list[dict[str, Any]],
    mjx_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "active": _force_semantics_metric_summary(
            baseline_rows=baseline_rows,
            mjx_rows=mjx_rows,
            replay_rows=replay_rows,
            metric="contact_force_active_mean",
        ),
        "peak": _force_semantics_metric_summary(
            baseline_rows=baseline_rows,
            mjx_rows=mjx_rows,
            replay_rows=replay_rows,
            metric="contact_force_peak",
        ),
    }


def _force_semantics_metric_summary(
    *,
    baseline_rows: list[dict[str, Any]],
    mjx_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    mjx_ratios: list[float] = []
    replay_ratios: list[float] = []
    for seed in SEEDS:
        baseline_value = _metric_float_for_seed(
            baseline_rows,
            seed=seed,
            metric=metric,
        )
        mjx_value = _metric_float_for_seed(mjx_rows, seed=seed, metric=metric)
        replay_value = _metric_float_for_seed(
            replay_rows,
            seed=seed,
            metric=metric,
        )
        mjx_ratio = _ratio(mjx_value, baseline_value)
        replay_ratio = _ratio(replay_value, baseline_value)
        if mjx_ratio is not None:
            mjx_ratios.append(mjx_ratio)
        if replay_ratio is not None:
            replay_ratios.append(replay_ratio)
        rows.append(
            {
                "seed": seed,
                "baseline": baseline_value,
                "mjx": mjx_value,
                "replay": replay_value,
                "mjx_to_baseline": mjx_ratio,
                "replay_to_baseline": replay_ratio,
                "classification": _force_ratio_classification(
                    mjx_ratio=mjx_ratio,
                    replay_ratio=replay_ratio,
                ),
                "semantic_classification": _force_semantics_classification(
                    mjx_ratio=mjx_ratio,
                    replay_ratio=replay_ratio,
                ),
            }
        )
    semantic_counts = _classification_counts(rows, "semantic_classification")
    return {
        "rows": rows,
        "mjx_to_baseline": _timing_stats(mjx_ratios),
        "replay_to_baseline": _timing_stats(replay_ratios),
        "force_ratio_high_count": sum(
            1 for row in rows if row["classification"] == "force_ratio_high"
        ),
        "semantic_classification_counts": semantic_counts,
        "isolated_raw_mjx_force_high_count": semantic_counts.get(
            "isolated_raw_mjx_force_high",
            0,
        ),
        "shared_force_high_count": semantic_counts.get("shared_force_high", 0),
        "replay_force_high_count": sum(
            1
            for ratio in replay_ratios
            if ratio > FORCE_REPLAY_NEAR_BASELINE_RATIO_MAX
        ),
    }


def _metric_float_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _metric_float(row, metric)
        if value is not None:
            values.append(value)
    return values


def _metric_float_for_seed(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    metric: str,
) -> float | None:
    for row in rows:
        if _safe_int(row.get("seed")) == seed:
            return _metric_float(row, metric)
    return None


def _metric_float(row: dict[str, Any], metric: str) -> float | None:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _force_ratio_classification(
    *,
    mjx_ratio: float | None,
    replay_ratio: float | None,
) -> str:
    if mjx_ratio is None or replay_ratio is None:
        return "force_ratio_missing"
    if mjx_ratio >= 3.0 and replay_ratio <= 1.5:
        return "force_ratio_high"
    return "force_ratio_not_isolated"


FORCE_MJX_HIGH_RATIO_MIN = 3.0
FORCE_REPLAY_NEAR_BASELINE_RATIO_MAX = 1.5


def _force_semantics_classification(
    *,
    mjx_ratio: float | None,
    replay_ratio: float | None,
) -> str:
    if mjx_ratio is None or replay_ratio is None:
        return "force_ratio_missing"
    if (
        mjx_ratio >= FORCE_MJX_HIGH_RATIO_MIN
        and replay_ratio <= FORCE_REPLAY_NEAR_BASELINE_RATIO_MAX
    ):
        return "isolated_raw_mjx_force_high"
    if (
        mjx_ratio >= FORCE_MJX_HIGH_RATIO_MIN
        and replay_ratio > FORCE_REPLAY_NEAR_BASELINE_RATIO_MAX
    ):
        return "shared_force_high"
    if replay_ratio > FORCE_REPLAY_NEAR_BASELINE_RATIO_MAX:
        return "replay_force_high"
    return "force_ratio_nominal"


def _classification_counts(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str):
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _timing_values(rows: list[dict[str, Any]], name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _timing_value(row, name)
        if _valid_timing(value):
            values.append(float(value))
    return values


def _window_count_values(rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _safe_int(row.get("num_windows", row.get("accepted_windows")))
        if value is not None and value > 0:
            values.append(float(value))
    return values


def _per_window_timing_values(rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        steady = _timing_value(row, "steady_state_wall_time_sec")
        windows = _safe_int(row.get("num_windows", row.get("accepted_windows")))
        if _valid_timing(steady) and windows is not None and windows > 0:
            values.append(float(steady) / float(windows))
    return values


def _timing_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "values": [],
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "values": values,
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def _classify_report(
    *,
    motion_results: dict[str, dict[str, Any]],
    replay_results: dict[str, dict[str, Any]],
    speed_results: dict[str, dict[str, Any]],
    realtime_results: dict[str, dict[str, Any]],
    target: str,
    passed: bool,
) -> str:
    if passed:
        return (
            "pass_4090_realtime"
            if target == TARGET_4090_REALTIME
            else "pass_h100_milestone"
        )
    if _has_invalid_benchmark_failure(
        motion_results,
        replay_results,
        speed_results,
        realtime_results,
    ):
        return "invalid_benchmark"
    if any(not result.get("passed") for result in replay_results.values()):
        return "parity_failure"
    if any(
        not result.get("baseline_passed") or not result.get("mjx_passed")
        for result in motion_results.values()
    ):
        return "quality_regression"
    target_results = (
        (speed_results, realtime_results)
        if target == TARGET_4090_REALTIME
        else (speed_results,)
    )
    if any(
        not result.get("passed")
        for result_group in target_results
        for result in result_group.values()
    ):
        return "speed_regression"
    return "invalid_benchmark"


def _replay_quality_failures(
    motion: str,
    rows: list[dict[str, Any]],
    baseline_envelope: dict[str, dict[str, float]],
) -> tuple[str, ...]:
    if len(rows) != len(SEEDS):
        return ()
    quality_rows = [_replay_quality_row(row) for row in rows]
    gate = evaluate_mjx_group(
        motion,
        quality_rows,
        baseline_envelope,
        MjxQualityPolicy.for_motion(motion),
    )
    return tuple(
        failure
        for failure in gate.failures
        if failure
        not in {
            "accepted_windows",
            "baseline_fallback",
            "mpc_accepted",
            "mpc_command_npz",
        }
    )


def _replay_quality_row(row: dict[str, Any]) -> dict[str, Any]:
    quality_row = dict(row)
    artifacts = dict(quality_row.get("artifacts", {}) or {})
    artifacts.setdefault("mpc_command_npz", "replay_uses_mjx_command")
    quality_row.update(
        {
            "mpc_accepted": True,
            "accepted_windows": 40,
            "mpc_used_baseline_fallback": False,
            "artifacts": artifacts,
        }
    )
    return quality_row


def _acceptance_row_metrics_provenance_failures(
    rows: list[dict[str, Any]],
    *,
    kind: str,
) -> tuple[str, ...]:
    if kind not in {"mjx", "replay"}:
        raise ValueError(f"Unsupported acceptance row kind: {kind!r}")
    argv_field = "mjx_argv" if kind == "mjx" else "replay_argv"
    failure = f"{kind}_metrics_provenance"
    failures: list[str] = []
    for row in rows:
        argv = row.get(argv_field)
        if not isinstance(argv, list):
            failures.append(failure)
            continue
        if (
            kind == "mjx"
            and "--mjx-enable-scan" in argv
            and _row_value(row, "physics_scan_enabled") is not True
        ):
            failures.append("mjx_physics_scan_enabled")
        if (
            kind == "mjx"
            and "--mjx-enable-scan" in argv
            and _mjx_physics_step_count_failure(row, argv)
        ):
            failures.append("mjx_physics_step_count")
        if not _acceptance_metrics_provenance_matches_argv(
            argv,
            kind=kind,
            parsed=row,
        ):
            failures.append(failure)
    return _unique(failures)


def _mjx_physics_step_count_failure(row: dict[str, Any], argv: list[str]) -> bool:
    expected_horizon = _safe_int(_argv_value(argv, "--mpc-planning-horizon-steps"))
    if expected_horizon is None or expected_horizon <= 0:
        return True
    count_min = _safe_int(_row_value(row, "physics_step_count_min"))
    if count_min is None or count_min < expected_horizon:
        return True
    count_windows = _safe_int(_row_value(row, "physics_step_count_windows"))
    accepted_windows = _safe_int(_row_value(row, "accepted_windows"))
    return (
        count_windows is None
        or accepted_windows is None
        or count_windows != accepted_windows
    )


def _replay_provenance_failures(
    rows: list[dict[str, Any]],
    mjx_rows: list[dict[str, Any]],
) -> tuple[str, ...]:
    failures: list[str] = []
    command_by_seed = _mjx_command_path_by_seed(mjx_rows)
    command_hash_by_seed = _mjx_command_hash_by_seed(mjx_rows)
    for row in rows:
        mpc = row.get("mpc")
        if not isinstance(mpc, dict):
            failures.append("replay_mode")
            failures.append("replay_saved_command")
            continue

        if mpc.get("replay_mode") != "shared_execute_backend":
            failures.append("replay_mode")
        if mpc.get("saved_command_replay_source") != "window_command_chunks":
            failures.append("replay_window_command_source")
        if row.get("metrics_method") != "replay_command":
            failures.append("replay_metrics_provenance")

        expected_saved = _argv_value(row.get("replay_argv", []), "--saved-command")
        saved = mpc.get("saved_command")
        if not _same_path(saved if isinstance(saved, str) else None, expected_saved):
            failures.append("replay_saved_command")
        elif not _is_existing_file(saved):
            failures.append("replay_saved_command")
        seed = _safe_int(row.get("seed"))
        source_command = command_by_seed.get(seed)
        if not _same_path(saved if isinstance(saved, str) else None, source_command):
            failures.append("replay_saved_command_source")
        source_hash = command_hash_by_seed.get(seed)
        saved_hash = mpc.get("saved_command_sha256")
        if saved_hash != source_hash:
            failures.append("replay_saved_command_hash")

        expected_control = _safe_int(
            _argv_value(row.get("replay_argv", []), "--replay-control-steps")
        )
        if _safe_int(mpc.get("control_steps")) != expected_control:
            failures.append("replay_control_steps")

        replay_steps = _safe_int(mpc.get("num_replay_steps"))
        if replay_steps != _safe_int(row.get("num_steps")):
            failures.append("replay_num_replay_steps")
        command_frames = _safe_int(mpc.get("num_command_frames"))
        if (
            command_frames is None
            or replay_steps is None
            or command_frames < replay_steps + 1
        ):
            failures.append("replay_num_command_frames")
        command_npz_frames = (
            _npz_frame_count(Path(saved).expanduser(), keys=("refined_qpos",))
            if isinstance(saved, str)
            else None
        )
        if command_frames is not None and command_npz_frames != command_frames:
            failures.append("replay_command_npz_frames")
        artifacts = row.get("artifacts", {})
        rollout_path = (
            artifacts.get("rollout_npz") if isinstance(artifacts, dict) else None
        )
        if (
            command_frames is not None
            and isinstance(rollout_path, str)
            and Path(rollout_path).expanduser().is_file()
            and not _replay_rollout_ref_indices_within_command(
                Path(rollout_path).expanduser(),
                command_frames=command_frames,
            )
        ):
            failures.append("replay_rollout_ref_indices")
    return _unique(failures)


def _mjx_command_path_by_seed(rows: list[dict[str, Any]]) -> dict[int, str]:
    command_by_seed: dict[int, str] = {}
    for row in rows:
        seed = _safe_int(row.get("seed"))
        artifacts = row.get("artifacts", {})
        if seed is None or not isinstance(artifacts, dict):
            continue
        command_path = artifacts.get("mpc_command_npz")
        if isinstance(command_path, str) and command_path.strip():
            command_by_seed[int(seed)] = command_path
    return command_by_seed


def _mjx_command_hash_by_seed(rows: list[dict[str, Any]]) -> dict[int, str]:
    hash_by_seed: dict[int, str] = {}
    for row in rows:
        seed = _safe_int(row.get("seed"))
        artifact_hashes = row.get("artifact_sha256", {})
        if seed is None or not isinstance(artifact_hashes, dict):
            continue
        command_hash = artifact_hashes.get("mpc_command_npz")
        if isinstance(command_hash, str) and command_hash.strip():
            hash_by_seed[int(seed)] = command_hash
    return hash_by_seed


def _npz_frame_count(path: Path, *, keys: tuple[str, ...]) -> int | None:
    try:
        with np.load(path) as data:
            for key in keys:
                if key in data.files:
                    shape = tuple(np.asarray(data[key]).shape)
                    if shape:
                        return int(shape[0])
    except Exception:
        return None
    return None


def _replay_rollout_ref_indices_within_command(
    path: Path,
    *,
    command_frames: int,
) -> bool:
    try:
        with np.load(path) as data:
            ref_indices = np.asarray(data["ref_indices"])
    except Exception:
        return False
    if ref_indices.size == 0:
        return False
    try:
        return bool(
            np.all(ref_indices >= 0)
            and np.all(ref_indices < int(command_frames))
        )
    except Exception:
        return False


def _has_invalid_benchmark_failure(
    motion_results: dict[str, dict[str, Any]],
    replay_results: dict[str, dict[str, Any]],
    speed_results: dict[str, dict[str, Any]],
    realtime_results: dict[str, dict[str, Any]],
) -> bool:
    invalid_markers = {
        "accepted_windows",
        "baseline_fallback",
        "baseline_required_gpu",
        "baseline_runtime_gpu_name",
        "baseline_runtime_visible_devices",
        "baseline_single_visible_gpu",
        "baseline_wall_time",
        "compile_init_wall_time",
        "contact_saturation",
        "fallback",
        "metrics_json",
        "metrics_json_hash",
        "metrics_json_stale",
        "max_contact_points_saturation",
        "max_geom_pairs_saturation",
        "mjx_compile_init_wall_time",
        "mjx_contact_diagnostics",
        "mjx_jit_warmup_enabled",
        "mjx_jit_warmup_wall_time",
        "mjx_metrics_provenance",
        "mjx_dynamic_execute_trace",
        "mjx_physics_scan_enabled",
        "mjx_physics_step_count",
        "mjx_runtime_visible_devices",
        "mjx_runtime_gpu_name",
        "mjx_required_gpu",
        "mjx_single_visible_gpu",
        "mjx_steady_state_wall_time",
        "control_dt_sec",
        "evaluated_motion_duration_sec",
        "motion_duration_sec",
        "mpc_accepted",
        "mpc_command_npz",
        "mpc_command_npz_hash",
        "mpc_command_qpos_mismatch",
        "mpc_command_npz_schema",
        "mpc_command_npz_stale",
        "mpc_rollout_qpos_mismatch",
        "num_steps",
        "repeat_count",
        "replay_control_steps",
        "replay_command_npz_frames",
        "replay_mode",
        "replay_metrics_provenance",
        "replay_num_command_frames",
        "replay_num_replay_steps",
        "replay_required_gpu",
        "replay_runtime_gpu_name",
        "replay_runtime_visible_devices",
        "replay_saved_command",
        "replay_saved_command_hash",
        "replay_saved_command_source",
        "replay_window_command_source",
        "replay_rollout_ref_indices",
        "replay_single_visible_gpu",
        "returncode",
        "rollout_npz",
        "rollout_npz_hash",
        "rollout_npz_schema",
        "rollout_npz_stale",
        "seed",
        "status",
    }
    for result in motion_results.values():
        failures = (*result.get("baseline_failures", ()), *result.get("mjx_failures", ()))
        if any(failure in invalid_markers for failure in failures):
            return True
    for result in replay_results.values():
        if any(failure in invalid_markers for failure in result.get("failures", ())):
            return True
    for result in speed_results.values():
        failures = result.get("failures", ())
        if any(failure in invalid_markers for failure in failures):
            return True
    for result in realtime_results.values():
        failures = result.get("failures", ())
        if any(failure in invalid_markers for failure in failures):
            return True
    return False


def _speed_gate_for_motion(
    baseline_rows: list[dict[str, Any]],
    mjx_rows: list[dict[str, Any]],
    *,
    min_speedup: float,
):
    baseline_time, baseline_failures = _mean_timing_strict(
        baseline_rows,
        "steady_state_wall_time_sec",
        expected_count=len(SEEDS),
        failure="baseline_wall_time",
    )
    mjx_time, mjx_failures = _mean_timing_strict(
        mjx_rows,
        "steady_state_wall_time_sec",
        expected_count=len(SEEDS),
        failure="mjx_steady_state_wall_time",
    )
    gate = evaluate_speed_gate(
        baseline_wall_time_sec=baseline_time,
        mjx_steady_state_wall_time_sec=mjx_time,
        min_speedup=min_speedup,
    )
    worst_speedup, worst_failures = _worst_seed_speedup(
        baseline_rows,
        mjx_rows,
        min_speedup=min_speedup,
    )
    failures = _unique(
        (*baseline_failures, *mjx_failures, *gate.failures, *worst_failures)
    )
    if failures:
        return SpeedGateResult(False, gate.speedup, failures, worst_speedup)
    return SpeedGateResult(gate.passed, gate.speedup, gate.failures, worst_speedup)


def _worst_seed_speedup(
    baseline_rows: list[dict[str, Any]],
    mjx_rows: list[dict[str, Any]],
    *,
    min_speedup: float,
) -> tuple[float | None, tuple[str, ...]]:
    baseline_times = _timing_by_seed(baseline_rows, "steady_state_wall_time_sec")
    mjx_times = _timing_by_seed(mjx_rows, "steady_state_wall_time_sec")
    if set(baseline_times) != set(SEEDS) or set(mjx_times) != set(SEEDS):
        return None, ()
    speedups = [
        baseline_times[seed] / mjx_times[seed]
        for seed in SEEDS
    ]
    worst_speedup = min(speedups)
    failures = ("speedup_worst",) if worst_speedup < float(min_speedup) else ()
    return worst_speedup, failures


def _timing_by_seed(
    rows: list[dict[str, Any]],
    name: str,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in rows:
        seed = _safe_int(row.get("seed"))
        value = _timing_value(row, name)
        if seed in SEEDS and _valid_timing(value) and float(value) > 0.0:
            values[int(seed)] = float(value)
    return values


def _realtime_gate_for_motion(
    mjx_rows: list[dict[str, Any]],
    *,
    min_realtime_factor: float,
    require_explicit_duration: bool = False,
) -> dict[str, Any]:
    duration, duration_failures = _mean_motion_duration_strict(
        mjx_rows,
        expected_count=len(SEEDS),
        require_explicit_duration=require_explicit_duration,
    )
    mjx_time, mjx_failures = _mean_timing_strict(
        mjx_rows,
        "steady_state_wall_time_sec",
        expected_count=len(SEEDS),
        failure="mjx_steady_state_wall_time",
    )
    failures = list(_unique((*duration_failures, *mjx_failures)))
    real_time_factor = float("nan")
    if not failures:
        real_time_factor = duration / mjx_time
        if real_time_factor < float(min_realtime_factor):
            failures.append("real_time_factor")
    return {
        "passed": not failures,
        "motion_duration_sec": duration,
        "mjx_steady_state_wall_time_sec": mjx_time,
        "real_time_factor": real_time_factor,
        "min_realtime_factor": float(min_realtime_factor),
        "failures": _unique(failures),
    }


def _mean_motion_duration_strict(
    rows: list[dict[str, Any]],
    *,
    expected_count: int,
    require_explicit_duration: bool,
) -> tuple[float, tuple[str, ...]]:
    values = []
    failures: list[str] = []
    for row in rows:
        value, row_failures = _motion_duration_sec(
            row,
            require_explicit_duration=require_explicit_duration,
        )
        failures.extend(row_failures)
        if (
            not row_failures
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0.0
        ):
            values.append(float(value))
        else:
            failures.append("motion_duration_sec")
    if len(values) != int(expected_count):
        failures.append("motion_duration_sec")
    if failures:
        return float("nan"), _unique(failures)
    return sum(values) / len(values), ()


def _motion_duration_sec(
    row: dict[str, Any],
    *,
    require_explicit_duration: bool = False,
) -> tuple[float | None, tuple[str, ...]]:
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    control_dt = _positive_float(
        row.get("control_dt_sec", metrics.get("control_dt_sec"))
    )
    evaluated_duration = _positive_float(
        row.get(
            "evaluated_motion_duration_sec",
            metrics.get("evaluated_motion_duration_sec"),
        )
    )
    steps = _safe_int(row.get("num_steps", metrics.get("num_steps")))

    failures: list[str] = []
    if require_explicit_duration:
        if control_dt is None:
            failures.append("control_dt_sec")
        if evaluated_duration is None:
            failures.append("evaluated_motion_duration_sec")
        if steps is None or steps <= 0:
            failures.append("num_steps")
        if (
            control_dt is not None
            and evaluated_duration is not None
            and steps is not None
            and steps > 0
        ):
            expected = float(steps) * float(control_dt)
            if not math.isclose(
                evaluated_duration,
                expected,
                rel_tol=1.0e-6,
                abs_tol=1.0e-6,
            ):
                failures.append("evaluated_motion_duration_sec")
        if failures:
            return None, _unique(failures)
        return evaluated_duration, ()

    if evaluated_duration is not None:
        return evaluated_duration, ()
    for candidate in (
        row.get("motion_duration_sec"),
        metrics.get("motion_duration_sec"),
        metrics.get("duration_sec"),
    ):
        value = _positive_float(candidate)
        if value is not None:
            return value, ()
    steps = _safe_int(row.get("num_steps", metrics.get("num_steps")))
    if steps is None or steps <= 0:
        return None, ("motion_duration_sec",)
    return float(steps) * float(POLICY_DT), ()


def _positive_float(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    ):
        return float(value)
    return None


def _mean_timing_strict(
    rows: list[dict[str, Any]],
    name: str,
    *,
    expected_count: int,
    failure: str,
) -> tuple[float, tuple[str, ...]]:
    values = []
    failures: list[str] = []
    for row in rows:
        value = _timing_value(row, name)
        if (
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0.0
        ):
            values.append(float(value))
        else:
            failures.append(failure)
    if len(values) != int(expected_count):
        failures.append(failure)
    if failures:
        return float("nan"), _unique(failures)
    return sum(values) / len(values), ()


def _row_from_metrics(metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text())
    if not isinstance(payload, dict):
        payload = {}
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    mpc = payload.get("mpc", {})
    if not isinstance(mpc, dict):
        mpc = {}
    accepted_windows = _strict_json_int(mpc.get("accepted_windows"))
    return {
        "metrics": metrics,
        "mpc": mpc,
        "metrics_method": payload.get("method"),
        "metrics_motion": payload.get("motion"),
        "metrics_device": payload.get("device"),
        "metrics_checkpoint": payload.get("checkpoint"),
        "metrics_max_steps": payload.get("max_steps"),
        "collision_profile": mpc.get(
            "collision_profile",
            payload.get("collision_profile"),
        ),
        "mpc_accepted": mpc.get("accepted") is True,
        "accepted_windows": -1 if accepted_windows is None else accepted_windows,
        "mpc_used_baseline_fallback": _strict_json_bool(
            mpc.get("used_baseline_fallback")
        ),
        "num_steps": _safe_int(metrics.get("num_steps", -1)),
        "compile_init_wall_time_sec": mpc.get("compile_init_wall_time_sec"),
        "jit_warmup_enabled": mpc.get("jit_warmup_enabled"),
        "jit_warmup_wall_time_sec": mpc.get("jit_warmup_wall_time_sec"),
        "optimizer_result_sync_wall_time_sec": mpc.get(
            "optimizer_result_sync_wall_time_sec"
        ),
        "physics_scan_enabled": mpc.get("physics_scan_enabled"),
        "mjx_impl": mpc.get("mjx_impl"),
        "mjx_model_impl": mpc.get("mjx_model_impl"),
        "mjx_model_options": mpc.get("mjx_model_options"),
        "mjx_warp_naconmax": mpc.get("mjx_warp_naconmax"),
        "mjx_warp_njmax": mpc.get("mjx_warp_njmax"),
        "rollout_source": mpc.get("rollout_source"),
        "rollout_dynamic_execute_trace": mpc.get("rollout_dynamic_execute_trace"),
        "execute_trace_chunks": mpc.get("execute_trace_chunks"),
        "execute_trace_source_counts": mpc.get("execute_trace_source_counts"),
        "strip_live_mjx_data_between_windows": mpc.get(
            "strip_live_mjx_data_between_windows"
        ),
        "score_only_optimizer": mpc.get("score_only_optimizer"),
        "contact_force_mode": mpc.get("contact_force_mode"),
        "contact_force_first_row_diagnostics": mpc.get(
            "contact_force_first_row_diagnostics"
        ),
        "contact_force_top_row_diagnostics": mpc.get(
            "contact_force_top_row_diagnostics"
        ),
        "current_controls_selected_windows": mpc.get(
            "current_controls_selected_windows"
        ),
        "noop_accepted_windows": mpc.get("noop_accepted_windows"),
        "zero_delta_noop_iteration_sum": mpc.get("zero_delta_noop_iteration_sum"),
        "zero_delta_noop_windows": mpc.get("zero_delta_noop_windows"),
        "zero_delta_noop_accepted_windows": mpc.get(
            "zero_delta_noop_accepted_windows"
        ),
        "score_threshold_noop_iteration_sum": mpc.get(
            "score_threshold_noop_iteration_sum"
        ),
        "score_threshold_noop_windows": mpc.get("score_threshold_noop_windows"),
        "score_threshold_noop_accepted_windows": mpc.get(
            "score_threshold_noop_accepted_windows"
        ),
        "top_score_gap_noop_iteration_sum": mpc.get(
            "top_score_gap_noop_iteration_sum"
        ),
        "top_score_gap_noop_windows": mpc.get("top_score_gap_noop_windows"),
        "top_score_gap_noop_accepted_windows": mpc.get(
            "top_score_gap_noop_accepted_windows"
        ),
        "control_delta_guard_noop_iteration_sum": mpc.get(
            "control_delta_guard_noop_iteration_sum"
        ),
        "control_delta_guard_noop_windows": mpc.get(
            "control_delta_guard_noop_windows"
        ),
        "control_delta_guard_noop_accepted_windows": mpc.get(
            "control_delta_guard_noop_accepted_windows"
        ),
        "noop_candidate_iteration_sum": mpc.get("noop_candidate_iteration_sum"),
        "noop_candidate_windows": mpc.get("noop_candidate_windows"),
        "noop_candidate_accepted_windows": mpc.get(
            "noop_candidate_accepted_windows"
        ),
        "accepted_iteration_sum": mpc.get("accepted_iteration_sum"),
        "iteration_accepted_window_counts": mpc.get(
            "iteration_accepted_window_counts"
        ),
        "iteration_current_selected_window_counts": mpc.get(
            "iteration_current_selected_window_counts"
        ),
        "iteration_noop_window_counts": mpc.get("iteration_noop_window_counts"),
        "iteration_zero_delta_noop_window_counts": mpc.get(
            "iteration_zero_delta_noop_window_counts"
        ),
        "iteration_score_threshold_noop_window_counts": mpc.get(
            "iteration_score_threshold_noop_window_counts"
        ),
        "iteration_control_delta_guard_noop_window_counts": mpc.get(
            "iteration_control_delta_guard_noop_window_counts"
        ),
        "iteration_noop_candidate_window_counts": mpc.get(
            "iteration_noop_candidate_window_counts"
        ),
        "top_score_gap_min": mpc.get("top_score_gap_min"),
        "top_score_gap_mean": mpc.get("top_score_gap_mean"),
        "top_score_gap_max": mpc.get("top_score_gap_max"),
        "top_score_gap_windows": mpc.get("top_score_gap_windows"),
        "iteration_top_score_gap_mins": mpc.get("iteration_top_score_gap_mins"),
        "iteration_top_score_gap_means": mpc.get("iteration_top_score_gap_means"),
        "sample_count": mpc.get("sample_count"),
        "optimizer_iterations": mpc.get("optimizer_iterations"),
        "planning_horizon_steps": mpc.get("planning_horizon_steps"),
        "control_steps": mpc.get("control_steps"),
        "knot_count": mpc.get("knot_count"),
        "temperature": mpc.get("temperature"),
        "elite_frac": mpc.get("elite_frac"),
        "root_pos_sigma": mpc.get("root_pos_sigma"),
        "root_rot_sigma": mpc.get("root_rot_sigma"),
        "joint_sigma": mpc.get("joint_sigma"),
        "first_ctrl_noise_scale": mpc.get("first_ctrl_noise_scale"),
        "last_ctrl_noise_scale": mpc.get("last_ctrl_noise_scale"),
        "final_noise_scale": mpc.get("final_noise_scale"),
        "sigma_decay": mpc.get("sigma_decay"),
        "mjx_min_score_improvement": mpc.get("mjx_min_score_improvement"),
        "mjx_min_top_score_gap": mpc.get("mjx_min_top_score_gap"),
        "mjx_cem_update_min_top_score_gap": mpc.get(
            "mjx_cem_update_min_top_score_gap"
        ),
        "mjx_max_control_delta": mpc.get("mjx_max_control_delta"),
        "candidate_rank_diagnostics_top_k": mpc.get(
            "candidate_rank_diagnostics_top_k",
            mpc.get("mjx_candidate_rank_diagnostics_top_k"),
        ),
        "candidate_rank_diagnostics_windows": mpc.get(
            "candidate_rank_diagnostics_windows"
        ),
        "mjx_candidate_rescore_diagnostics": mpc.get(
            "mjx_candidate_rescore_diagnostics"
        ),
        "candidate_rescore_diagnostics_windows": mpc.get(
            "candidate_rescore_diagnostics_windows"
        ),
        "candidate_rescore_score_delta_max": mpc.get(
            "candidate_rescore_score_delta_max"
        ),
        "candidate_rescore_score_delta_mean": mpc.get(
            "candidate_rescore_score_delta_mean"
        ),
        "candidate_rescore_top1_changed_iteration_sum": mpc.get(
            "candidate_rescore_top1_changed_iteration_sum"
        ),
        "candidate_rescore_selection_top_k": mpc.get(
            "candidate_rescore_selection_top_k",
            mpc.get("mjx_candidate_rescore_selection_top_k"),
        ),
        "candidate_rescore_selection_windows": mpc.get(
            "candidate_rescore_selection_windows"
        ),
        "candidate_rescore_selection_score_delta_max": mpc.get(
            "candidate_rescore_selection_score_delta_max"
        ),
        "candidate_rescore_selection_score_delta_mean": mpc.get(
            "candidate_rescore_selection_score_delta_mean"
        ),
        "candidate_rescore_selection_changed_iteration_sum": mpc.get(
            "candidate_rescore_selection_changed_iteration_sum"
        ),
        "candidate_score_component_diagnostics_top_k": mpc.get(
            "candidate_score_component_diagnostics_top_k",
            mpc.get("mjx_candidate_score_component_diagnostics_top_k"),
        ),
        "use_warm_start": mpc.get("use_warm_start"),
        "use_guided_candidate": mpc.get("use_guided_candidate"),
        "guided_candidate_period": mpc.get("guided_candidate_period"),
        "guided_candidate_windows": mpc.get("guided_candidate_windows"),
        "physics_step_count_min": mpc.get("physics_step_count_min"),
        "physics_step_count_max": mpc.get("physics_step_count_max"),
        "physics_step_count_windows": mpc.get("physics_step_count_windows"),
        "runtime_visible_devices": mpc.get("runtime_visible_devices"),
        "runtime_gpu_name": mpc.get("runtime_gpu_name"),
        "steady_state_wall_time_sec": mpc.get("steady_state_wall_time_sec"),
        "control_dt_sec": metrics.get("control_dt_sec"),
        "evaluated_motion_duration_sec": metrics.get("evaluated_motion_duration_sec"),
        "contact_saturated": mpc.get("contact_saturated"),
        "max_contact_points_saturated": mpc.get("max_contact_points_saturated"),
        "max_geom_pairs_saturated": mpc.get("max_geom_pairs_saturated"),
        "max_contact_points": mpc.get("max_contact_points"),
        "max_geom_pairs": mpc.get("max_geom_pairs"),
        "contact_pair_count": mpc.get("contact_pair_count"),
        "active_contact_count": mpc.get("active_contact_count"),
    }


def _attach_single_row_speed_evidence(
    row: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    *,
    min_speedup: float,
    min_realtime_factor: float,
) -> dict[str, Any]:
    row = dict(row)
    motion = row.get("motion")
    seed = _safe_int(row.get("seed"))
    mjx_time = _timing_value(row, "steady_state_wall_time_sec")
    baseline_time = None
    if isinstance(motion, str) and seed in SEEDS:
        for baseline_row in baseline_rows:
            if baseline_row.get("motion_name") != motion:
                continue
            if _safe_int(baseline_row.get("seed")) != seed:
                continue
            candidate = _timing_value(baseline_row, "steady_state_wall_time_sec")
            if _valid_timing(candidate):
                baseline_time = float(candidate)
                break
    if baseline_time is not None:
        row["same_seed_baseline_steady_state_wall_time_sec"] = baseline_time
        target_time = baseline_time / float(min_speedup)
        row["same_seed_target_steady_state_wall_time_sec"] = target_time
        if _valid_timing(mjx_time):
            mjx_time = float(mjx_time)
            speedup = baseline_time / mjx_time
            row["same_seed_speedup"] = speedup
            row["same_seed_speedup_passed"] = speedup >= float(min_speedup)
            row["same_seed_speedup_shortfall_sec"] = max(0.0, mjx_time - target_time)
    duration = _positive_float(row.get("evaluated_motion_duration_sec"))
    if duration is not None and _valid_timing(mjx_time):
        realtime_factor = duration / float(mjx_time)
        row["realtime_factor"] = realtime_factor
        row["realtime_passed"] = realtime_factor >= float(min_realtime_factor)
    return row


def _strict_json_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _strict_json_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _mjx_timing_evidence_failures(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    failures: list[str] = []
    for row in rows:
        if not _valid_timing(_timing_value(row, "compile_init_wall_time_sec")):
            failures.append("mjx_compile_init_wall_time")
        if row.get("jit_warmup_enabled") is not True:
            failures.append("mjx_jit_warmup_enabled")
        if not _valid_timing(_timing_value(row, "jit_warmup_wall_time_sec")):
            failures.append("mjx_jit_warmup_wall_time")
    return _unique(failures)


def _mjx_runtime_evidence_failures(
    rows: list[dict[str, Any]],
    *,
    required_gpu_name_fragment: str,
) -> tuple[str, ...]:
    return _runtime_evidence_failures(
        rows,
        prefix="mjx",
        required_gpu_name_fragment=required_gpu_name_fragment,
    )


def _mjx_contact_evidence_failures(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    failures: list[str] = []
    for row in rows:
        for field in MJX_CONTACT_SATURATION_FIELDS:
            if not isinstance(_row_value(row, field), bool):
                failures.append("mjx_contact_diagnostics")
        counts = {field: _row_value(row, field) for field in MJX_CONTACT_COUNT_FIELDS}
        for field, value in counts.items():
            if not _valid_contact_count(
                value,
                positive=field in {"max_contact_points", "max_geom_pairs"},
            ):
                failures.append("mjx_contact_diagnostics")
        if (
            _valid_contact_count(counts["active_contact_count"])
            and _valid_contact_count(counts["max_contact_points"], positive=True)
            and float(counts["active_contact_count"])
            > float(counts["max_contact_points"])
        ):
            failures.append("mjx_contact_diagnostics")
        if (
            _valid_contact_count(counts["contact_pair_count"])
            and _valid_contact_count(counts["max_geom_pairs"], positive=True)
            and float(counts["contact_pair_count"])
            > float(counts["max_geom_pairs"])
        ):
            failures.append("mjx_contact_diagnostics")
    return _unique(failures)


def _mjx_dynamic_trace_evidence_failures(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    failures: list[str] = []
    for row in rows:
        if row.get("rollout_source") != "dynamic_execute_trace":
            failures.append("mjx_dynamic_execute_trace")
        if row.get("rollout_dynamic_execute_trace") is not True:
            failures.append("mjx_dynamic_execute_trace")
        chunks = _safe_int(row.get("execute_trace_chunks"))
        windows = _safe_int(row.get("accepted_windows"))
        if chunks is None or chunks <= 0:
            failures.append("mjx_dynamic_execute_trace")
        if chunks is not None and windows is not None and chunks != windows:
            failures.append("mjx_dynamic_execute_trace")
        if _row_value(row, "mjx_impl") == "warp":
            counts = _execute_trace_source_counts(row)
            if counts is None:
                failures.append("mjx_dynamic_execute_trace")
            elif any(
                source != "rollout_tracer" and int(count) > 0
                for source, count in counts.items()
            ):
                failures.append("mjx_dynamic_execute_trace")
            elif (
                windows is not None
                and int(counts.get("rollout_tracer", 0)) != int(windows)
            ):
                failures.append("mjx_dynamic_execute_trace")
    return _unique(failures)


def _execute_trace_source_counts(row: dict[str, Any]) -> dict[str, int] | None:
    counts = _row_value(row, "execute_trace_source_counts")
    if isinstance(counts, dict):
        normalized: dict[str, int] = {}
        for source, value in counts.items():
            parsed = _safe_int(value)
            if parsed is None or parsed < 0:
                return None
            normalized[str(source)] = parsed
        return normalized
    mpc = row.get("mpc", {})
    history = mpc.get("history") if isinstance(mpc, dict) else None
    if not isinstance(history, list):
        history = row.get("history")
    if not isinstance(history, list):
        return None
    normalized = {}
    for info in history:
        if not isinstance(info, dict):
            continue
        source = info.get("execute_trace_source")
        if source is None:
            continue
        source_name = str(source)
        normalized[source_name] = normalized.get(source_name, 0) + 1
    return normalized if normalized else None


def _baseline_artifact_evidence_failures(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    failures: list[str] = []
    for row in rows:
        artifacts = row.get("artifacts", {})
        for key in REQUIRED_ARTIFACT_FIELDS:
            if not isinstance(artifacts, dict):
                failures.append(key)
                continue
            value = artifacts.get(key)
            if not isinstance(value, str) or not Path(value).expanduser().is_file():
                failures.append(key)
    return _unique(failures)


def _baseline_runtime_evidence_failures(
    rows: list[dict[str, Any]],
    *,
    required_gpu_name_fragment: str,
) -> tuple[str, ...]:
    return _runtime_evidence_failures(
        rows,
        prefix="baseline",
        required_gpu_name_fragment=required_gpu_name_fragment,
    )


def _runtime_evidence_failures(
    rows: list[dict[str, Any]],
    *,
    prefix: str,
    required_gpu_name_fragment: str,
) -> tuple[str, ...]:
    failures: list[str] = []
    required = str(required_gpu_name_fragment)
    for row in rows:
        devices = _row_value(row, "runtime_visible_devices")
        if not isinstance(devices, (list, tuple)) or not devices:
            failures.append(f"{prefix}_runtime_visible_devices")
        else:
            visible = tuple(str(value) for value in devices if str(value))
            if len(visible) != 1:
                failures.append(f"{prefix}_single_visible_gpu")
        gpu_name = _row_value(row, "runtime_gpu_name")
        if not isinstance(gpu_name, str) or not gpu_name.strip():
            failures.append(f"{prefix}_runtime_gpu_name")
        elif required and required not in gpu_name:
            failures.append(f"{prefix}_required_gpu")
    return _unique(failures)


def _baseline_artifact_preflight_failures(
    baseline_rows: list[dict[str, Any]],
) -> tuple[str, ...]:
    failures: list[str] = []
    for motion in MOTIONS:
        baseline_group = [
            row for row in baseline_rows if row.get("motion_name") == motion
        ]
        failures.extend(_baseline_artifact_failures(baseline_group))
    return _unique(failures)


def _baseline_artifact_failures(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    return _unique(
        (
            *_baseline_artifact_evidence_failures(rows),
            *_artifact_freshness_failures(
                rows,
                artifact_fields=REQUIRED_ARTIFACT_FIELDS,
            ),
            *_artifact_hash_failures(
                rows,
                artifact_fields=REQUIRED_ARTIFACT_FIELDS,
            ),
            *_artifact_npz_schema_failures(
                rows,
                require_command=True,
                require_rollout_command_match=False,
            ),
        )
    )


def _artifact_freshness_failures(
    rows: list[dict[str, Any]],
    *,
    artifact_fields: tuple[str, ...],
) -> tuple[str, ...]:
    failures: list[str] = []
    for row in rows:
        start_ns = _safe_int(row.get("command_start_time_ns"))
        if start_ns is None:
            if _requires_freshness_evidence(row):
                failures.extend(f"{key}_stale" for key in artifact_fields)
            continue
        mtimes = row.get("artifact_mtime_ns", {})
        if not isinstance(mtimes, dict):
            mtimes = {}
        for key in artifact_fields:
            mtime_ns = _safe_int(mtimes.get(key))
            if mtime_ns is None:
                if _requires_freshness_evidence(row):
                    failures.append(f"{key}_stale")
                continue
            if int(mtime_ns) + ARTIFACT_FRESHNESS_TOLERANCE_NS < int(start_ns):
                failures.append(f"{key}_stale")
    return _unique(failures)


def _requires_freshness_evidence(row: dict[str, Any]) -> bool:
    return bool(row.get("reused_existing")) or row.get("motion_name") in MOTIONS


def _artifact_hash_failures(
    rows: list[dict[str, Any]],
    *,
    artifact_fields: tuple[str, ...],
) -> tuple[str, ...]:
    failures: list[str] = []
    for row in rows:
        artifacts = row.get("artifacts", {})
        expected_hashes = row.get("artifact_sha256", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        if not isinstance(expected_hashes, dict):
            expected_hashes = {}
        for key in artifact_fields:
            expected = expected_hashes.get(key)
            if not isinstance(expected, str) or not expected.strip():
                failures.append(f"{key}_hash")
                continue
            artifact = artifacts.get(key)
            if not isinstance(artifact, str) or not Path(artifact).expanduser().is_file():
                continue
            actual = _file_sha256(Path(artifact).expanduser())
            if actual != expected:
                failures.append(f"{key}_hash")
    return _unique(failures)


def _artifact_npz_schema_failures(
    rows: list[dict[str, Any]],
    *,
    require_command: bool,
    require_rollout_command_match: bool = True,
) -> tuple[str, ...]:
    failures: list[str] = []
    for row in rows:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        num_steps = _safe_int(row.get("num_steps", metrics.get("num_steps")))
        artifacts = row.get("artifacts", {})
        if not isinstance(artifacts, dict) or num_steps is None:
            continue
        rollout_path = artifacts.get("rollout_npz")
        rollout_valid = False
        if isinstance(rollout_path, str) and Path(rollout_path).expanduser().is_file():
            rollout_valid = _valid_rollout_npz(
                Path(rollout_path).expanduser(),
                num_steps=num_steps,
            )
            if not rollout_valid:
                failures.append("rollout_npz_schema")
        if require_command:
            command_path = artifacts.get("mpc_command_npz")
            command_valid = False
            if isinstance(command_path, str) and Path(command_path).expanduser().is_file():
                command_valid = _valid_command_npz(
                    Path(command_path).expanduser(),
                    num_steps=num_steps,
                )
                if not command_valid:
                    failures.append("mpc_command_npz_schema")
                elif not _command_qpos_is_consistent(
                    Path(command_path).expanduser(),
                ):
                    failures.append("mpc_command_qpos_mismatch")
                elif not _command_qvel_is_consistent(
                    Path(command_path).expanduser(),
                ):
                    failures.append("mpc_command_qvel_mismatch")
            if (
                require_rollout_command_match
                and rollout_valid
                and command_valid
                and isinstance(rollout_path, str)
                and isinstance(command_path, str)
                and not _rollout_matches_command_npz(
                    Path(rollout_path).expanduser(),
                    Path(command_path).expanduser(),
                )
            ):
                failures.append("mpc_rollout_qpos_mismatch")
    return _unique(failures)


def _valid_rollout_npz(path: Path, *, num_steps: int) -> bool:
    frames = int(num_steps) + 1
    bodies = len(MUJOCO_BODY_NAMES)
    required_shapes = {
        "qpos": (frames, 1, QPOS_DIM),
        "qvel": (frames, 1, QVEL_DIM),
        "body_pos_w": (frames, 1, bodies, 3),
        "body_quat_w": (frames, 1, bodies, 4),
        "body_lin_vel_w": (frames, 1, bodies, 3),
        "body_ang_vel_w": (frames, 1, bodies, 3),
        "actions": (int(num_steps), 1, ACTION_DIM),
        "controls": (int(num_steps), 1, ACTION_DIM),
        "contact_indicator": (frames, 1, 2),
        "contact_force": (frames, 1, 2),
        "floor_contact_indicator": (frames, 1, 3),
        "floor_contact_force": (frames, 1, 3),
        "ref_indices": (frames, 1),
    }
    try:
        with np.load(path) as data:
            if "dt" not in data.files or np.asarray(data["dt"]).shape not in {(), (1,)}:
                return False
            if not _npz_has_shapes(data, required_shapes):
                return False
            ref_indices = np.asarray(data["ref_indices"]).reshape(-1)
            expected_refs = np.concatenate(
                (
                    np.array([0], dtype=ref_indices.dtype),
                    np.arange(frames - 1, dtype=ref_indices.dtype),
                )
            )
            return bool(np.array_equal(ref_indices, expected_refs))
    except Exception:
        return False


def _valid_command_npz(path: Path, *, num_steps: int) -> bool:
    frames = int(num_steps) + 1
    bodies = len(MUJOCO_BODY_NAMES)
    required_shapes = {
        "refined_qpos": ((frames, QPOS_DIM), (frames, 1, QPOS_DIM)),
        "candidate_scores": None,
        "command_joint_pos": (frames, 1, ACTION_DIM),
        "command_joint_vel": (frames, 1, ACTION_DIM),
        "command_body_pos_w": (frames, 1, bodies, 3),
        "command_body_quat_w": (frames, 1, bodies, 4),
        "command_body_lin_vel_w": (frames, 1, bodies, 3),
        "command_body_ang_vel_w": (frames, 1, bodies, 3),
        "command_qpos_trajectory": ((frames, QPOS_DIM), (frames, 1, QPOS_DIM)),
        "command_qvel_trajectory": ((frames, QVEL_DIM), (frames, 1, QVEL_DIM)),
    }
    try:
        with np.load(path) as data:
            return _npz_has_shapes(data, required_shapes) and _valid_window_command_chunks(
                data,
                num_steps=int(num_steps),
            )
    except Exception:
        return False


def _valid_window_command_chunks(
    data: np.lib.npyio.NpzFile,
    *,
    num_steps: int,
) -> bool:
    bodies = len(MUJOCO_BODY_NAMES)
    metadata = ("window_starts", "window_execute_steps", "window_horizons")
    if any(name not in data.files for name in metadata):
        return False
    starts = np.asarray(data["window_starts"])
    execute_steps = np.asarray(data["window_execute_steps"])
    horizons = np.asarray(data["window_horizons"])
    if starts.ndim != 1 or execute_steps.shape != starts.shape or horizons.shape != starts.shape:
        return False
    if starts.shape[0] < 1:
        return False
    starts = starts.astype(np.int64, copy=False)
    execute_steps = execute_steps.astype(np.int64, copy=False)
    horizons = horizons.astype(np.int64, copy=False)
    if int(starts[0]) != 0:
        return False
    if np.any(execute_steps < 1) or np.any(horizons < execute_steps):
        return False
    expected_starts = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(execute_steps[:-1])]
    )
    if not np.array_equal(starts, expected_starts):
        return False
    if int(starts[-1] + execute_steps[-1]) != int(num_steps):
        return False
    max_horizon = int(horizons.max())
    required = {
        "window_command_joint_pos": (starts.shape[0], max_horizon, 1, ACTION_DIM),
        "window_command_joint_vel": (starts.shape[0], max_horizon, 1, ACTION_DIM),
        "window_command_body_pos_w": (starts.shape[0], max_horizon, 1, bodies, 3),
        "window_command_body_quat_w": (starts.shape[0], max_horizon, 1, bodies, 4),
        "window_command_body_lin_vel_w": (starts.shape[0], max_horizon, 1, bodies, 3),
        "window_command_body_ang_vel_w": (starts.shape[0], max_horizon, 1, bodies, 3),
        "window_command_qpos_chunks": (starts.shape[0], max_horizon, 1, QPOS_DIM),
        "window_command_qvel_chunks": (starts.shape[0], max_horizon, 1, QVEL_DIM),
    }
    if "window_command_schema_version" not in data.files:
        return False
    return _npz_has_shapes(data, required)


def _rollout_matches_command_npz(rollout_path: Path, command_path: Path) -> bool:
    try:
        with np.load(rollout_path) as rollout, np.load(command_path) as command:
            rollout_qpos = np.asarray(rollout["qpos"])[:, 0]
            refined_qpos = _squeeze_single_batch_axis(command["refined_qpos"])
            if rollout_qpos.shape != refined_qpos.shape:
                return False
            return bool(
                np.allclose(
                    rollout_qpos,
                    refined_qpos,
                    atol=1.0e-5,
                    rtol=1.0e-5,
                )
            )
    except Exception:
        return False


def _command_qpos_is_consistent(command_path: Path) -> bool:
    try:
        with np.load(command_path) as command:
            refined_qpos = _squeeze_single_batch_axis(command["refined_qpos"])
            command_qpos = _squeeze_single_batch_axis(
                command["command_qpos_trajectory"]
            )
            if refined_qpos.shape != command_qpos.shape:
                return False
            return bool(
                np.allclose(
                    refined_qpos,
                    command_qpos,
                    atol=1.0e-5,
                    rtol=1.0e-5,
                )
            )
    except Exception:
        return False


def _command_qvel_is_consistent(command_path: Path) -> bool:
    try:
        with np.load(command_path) as command:
            refined_qpos = _squeeze_single_batch_axis(command["refined_qpos"])
            command_qvel = _squeeze_single_batch_axis(
                command["command_qvel_trajectory"]
            )
            expected_qvel = _qvel_from_qpos_trajectory(refined_qpos)
            if command_qvel.shape != expected_qvel.shape:
                return False
            return bool(
                np.allclose(
                    command_qvel,
                    expected_qvel,
                    atol=COMMAND_QVEL_CONSISTENCY_ATOL,
                    rtol=COMMAND_QVEL_CONSISTENCY_RTOL,
                )
            )
    except Exception:
        return False


def _qvel_from_qpos_trajectory(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[-1] != QPOS_DIM:
        raise ValueError(f"Expected qpos shape (frames, {QPOS_DIM}), got {qpos.shape}")
    qvel = np.zeros((qpos.shape[0], QVEL_DIM), dtype=np.float32)
    if qpos.shape[0] <= 1:
        return qvel
    lin_vel = _differentiate(qpos[:, :3])
    delta_quat = _quat_mul(qpos[1:, 3:7], _quat_inv(qpos[:-1, 3:7]))
    ang_vel = _axis_angle_from_quat(delta_quat) / POLICY_DT
    ang_vel = np.concatenate([ang_vel, ang_vel[-1:]], axis=0)
    qvel[:, :6] = _world_velocity_to_qvel(
        qpos[:, :7],
        np.concatenate([lin_vel, ang_vel], axis=-1),
    )
    qvel[:, 6:] = _differentiate(qpos[:, 7:])
    return qvel


def _differentiate(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    velocity = (values[1:] - values[:-1]) / POLICY_DT
    return np.concatenate([velocity, velocity[-1:]], axis=0).astype(np.float32)


def _quat_inv(quat: np.ndarray) -> np.ndarray:
    conjugate = np.concatenate([quat[..., 0:1], -quat[..., 1:]], axis=-1)
    return conjugate / np.sum(quat * quat, axis=-1, keepdims=True).clip(min=1.0e-9)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1, q2 = np.broadcast_arrays(q1, q2)
    w1, x1, y1, z1 = (q1[..., index] for index in range(4))
    w2, x2, y2, z2 = (q2[..., index] for index in range(4))
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z], axis=-1)


def _axis_angle_from_quat(quat: np.ndarray) -> np.ndarray:
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = np.linalg.norm(quat[..., 1:], axis=-1)
    half_angle = np.arctan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    denom = np.where(
        np.abs(angle) > 1.0e-6,
        np.sin(half_angle) / np.where(np.abs(angle) > 1.0e-6, angle, 1.0),
        0.5 - angle * angle / 48.0,
    )
    return quat[..., 1:4] / denom[..., None]


def _world_velocity_to_qvel(qpos: np.ndarray, world_vel: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            world_vel[..., :3],
            _quat_apply_inverse(qpos[..., 3:7], world_vel[..., 3:6]),
        ],
        axis=-1,
    )


def _quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    xyz = quat[..., 1:]
    t = np.cross(xyz, vec) * 2.0
    return vec - quat[..., 0:1] * t + np.cross(xyz, t)


def _squeeze_single_batch_axis(array) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 3 and value.shape[1] == 1:
        return value[:, 0]
    return value


def _npz_has_shapes(
    data,
    required_shapes: dict[str, tuple[int, ...] | tuple[tuple[int, ...], ...] | None],
) -> bool:
    for key, expected in required_shapes.items():
        if key not in data.files:
            return False
        if expected is None:
            if np.asarray(data[key]).size <= 0:
                return False
            continue
        observed = tuple(np.asarray(data[key]).shape)
        allowed = expected if expected and isinstance(expected[0], tuple) else (expected,)
        if observed not in allowed:
            return False
    return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_evidence_failures(
    rows: list[dict[str, Any]],
    *,
    timing_failure: str | None = None,
    timing_fields: tuple[str, ...] = ("steady_state_wall_time_sec",),
    artifact_fields: tuple[str, ...] = REQUIRED_ARTIFACT_FIELDS,
    require_mpc_fields: bool = True,
) -> tuple[str, ...]:
    failures: list[str] = []
    if len(rows) != len(SEEDS):
        failures.append("repeat_count")
    seen: set[int] = set()
    for row in rows:
        seed = _safe_int(row.get("seed"))
        if seed not in SEEDS:
            failures.append("seed")
        elif seed in seen:
            failures.append("seed")
        else:
            seen.add(seed)
        if row.get("status") != "ok":
            failures.append("status")
        if row.get("returncode") != 0:
            failures.append("returncode")
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            failures.append("metrics")
            metrics = {}
        for metric in ("success", "score", *PRIMARY_ERROR_METRICS):
            if not _has_valid_metric(metrics, metric):
                failures.append(f"{metric}_missing")
        if require_mpc_fields:
            if row.get("mpc_accepted") is not True:
                failures.append("mpc_accepted")
            if _safe_int(row.get("accepted_windows")) != 40:
                failures.append("accepted_windows")
            if row.get("mpc_used_baseline_fallback") is not False:
                failures.append("baseline_fallback")
        artifacts = row.get("artifacts", {})
        for key in artifact_fields:
            if not isinstance(artifacts, dict) or not artifacts.get(key):
                failures.append(key)
        if timing_failure is not None:
            if not any(_valid_timing(_timing_value(row, field)) for field in timing_fields):
                failures.append(timing_failure)
    missing = set(SEEDS) - seen
    if missing:
        failures.append("seed")
    return _unique(failures)


def _skip_report_row_evidence_failures(
    mjx_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    *,
    required_gpu_name_fragment: str,
) -> tuple[str, ...]:
    failures: list[str] = []
    for row in mjx_rows:
        failures.extend(
            _single_row_evidence_failures(
                row,
                artifact_fields=REQUIRED_ARTIFACT_FIELDS,
                timing_failure="mjx_steady_state_wall_time",
            )
        )
        if row.get("mpc_accepted") is not True:
            failures.append("mpc_accepted")
        if _safe_int(row.get("accepted_windows")) != 40:
            failures.append("accepted_windows")
        if row.get("mpc_used_baseline_fallback") is not False:
            failures.append("baseline_fallback")
        failures.extend(_mjx_dynamic_trace_evidence_failures([row]))
        if row.get("same_seed_speedup_passed") is False:
            failures.append("same_seed_speedup")
        elif "same_seed_speedup_passed" not in row:
            failures.append("same_seed_speedup")
    failures.extend(
        _mjx_runtime_evidence_failures(
            mjx_rows,
            required_gpu_name_fragment=required_gpu_name_fragment,
        )
    )
    for row in replay_rows:
        failures.extend(
            _single_row_evidence_failures(
                row,
                artifact_fields=("metrics_json", "rollout_npz"),
                timing_failure="replay_steady_state_wall_time",
                timing_fields=("command_wall_time_sec", "steady_state_wall_time_sec"),
            )
        )
    failures.extend(
        _runtime_evidence_failures(
            replay_rows,
            prefix="replay",
            required_gpu_name_fragment=required_gpu_name_fragment,
        )
    )
    return _unique(failures)


def _single_row_evidence_failures(
    row: dict[str, Any],
    *,
    artifact_fields: tuple[str, ...],
    timing_failure: str,
    timing_fields: tuple[str, ...] = ("steady_state_wall_time_sec",),
) -> tuple[str, ...]:
    failures: list[str] = []
    if row.get("status") != "ok":
        failures.append("status")
    if row.get("returncode") != 0:
        failures.append("returncode")
    if _safe_int(row.get("num_steps")) != 800:
        failures.append("num_steps")
    artifacts = row.get("artifacts", {})
    for key in artifact_fields:
        if not isinstance(artifacts, dict) or not artifacts.get(key):
            failures.append(key)
    if not any(_valid_timing(_timing_value(row, field)) for field in timing_fields):
        failures.append(timing_failure)
    return _unique(failures)


def _timing_value(row: dict[str, Any], name: str):
    return _row_value(row, name)


def _row_value(row: dict[str, Any], name: str):
    mpc = row.get("mpc", {})
    return row.get(name, mpc.get(name) if isinstance(mpc, dict) else None)


def _valid_timing(value) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _has_valid_metric(metrics: dict[str, Any], name: str) -> bool:
    value = metrics.get(name)
    if isinstance(value, bool):
        return True
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _contact_count_values(rows: list[dict[str, Any]], name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _row_value(row, name)
        if _valid_contact_count(value):
            values.append(float(value))
    return values


def _contact_flag_value(row: dict[str, Any], name: str) -> bool | None:
    value = _row_value(row, name)
    return value if isinstance(value, bool) else None


def _valid_contact_count(value, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(float(value)):
        return False
    lower_bound = 1.0 if positive else 0.0
    if float(value) < lower_bound:
        return False
    return float(value).is_integer()


def _safe_int(value) -> int | None:
    try:
        parsed = int(value)
        exact = float(parsed) == float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if exact else None


def _unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _attach_artifacts(row: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser()
    row = dict(row)
    artifact_paths = {
        "metrics_json": output_dir / "metrics.json",
        "rollout_npz": output_dir / "rollout.npz",
        "mpc_command_npz": output_dir / "mpc_command.npz",
    }
    row["artifacts"] = {
        key: _existing_path(path)
        for key, path in artifact_paths.items()
    }
    row["artifact_mtime_ns"] = {
        key: path.stat().st_mtime_ns if path.exists() else None
        for key, path in artifact_paths.items()
    }
    row["artifact_sha256"] = {
        key: _file_sha256(path)
        for key, path in artifact_paths.items()
        if path.exists()
    }
    return row


def _existing_path(path: Path) -> str | None:
    return str(path.resolve()) if path.exists() else None


def _output_dir_from_argv(argv: list[str]) -> Path:
    try:
        return Path(argv[argv.index("--output-dir") + 1]).expanduser()
    except (ValueError, IndexError) as exc:
        raise ValueError("Command argv is missing --output-dir") from exc


def _argv_value(argv: Any, flag: str) -> str | None:
    if not isinstance(argv, list):
        return None
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def _same_path(left: str | None, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _is_existing_file(path: str | None) -> bool:
    return isinstance(path, str) and Path(path).expanduser().is_file()


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _set_arg(argv: list[str], flag: str, value: str) -> list[str]:
    out = list(argv)
    try:
        index = out.index(flag)
    except ValueError:
        out.extend([flag, value])
    else:
        if index + 1 >= len(out):
            raise ValueError(f"{flag} has no value")
        out[index + 1] = value
    return out


def _drop_arg_with_value(argv: list[str], flag: str) -> list[str]:
    out = []
    skip_next = False
    for index, value in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if value == flag:
            if index + 1 >= len(argv):
                raise ValueError(f"{flag} has no value")
            skip_next = True
            continue
        out.append(value)
    return out


def _drop_flag(argv: list[str], flag: str) -> list[str]:
    return [value for value in argv if value != flag]


if __name__ == "__main__":
    raise SystemExit(main())
