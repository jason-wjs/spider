#!/usr/bin/env python3
"""Preflight the local RTX 4090 runtime before G1 WBC MJX experiments."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REQUIRED_IMPORTS = (
    "torch",
    "jax",
    "jax.numpy",
    "mujoco",
    "mujoco.mjx",
    "mujoco_warp",
    "warp",
)
NVIDIA_SMI_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,driver_version",
    "--format=csv,noheader",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check local CUDA/JAX/MJX runtime readiness for G1 WBC 4090 runs."
    )
    parser.add_argument(
        "--required-gpu-name-fragment",
        default="4090",
        help="Substring required in the detected JAX or nvidia-smi GPU name.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to write the preflight report JSON.",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(
    *,
    required_gpu_name_fragment: str,
    environ: dict[str, str] | None = None,
    import_module=importlib.import_module,
    run_command=None,
) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    run_command = run_nvidia_smi if run_command is None else run_command
    visible_devices = _visible_cuda_devices(env)
    import_status, modules = _probe_imports(import_module)
    jax_devices, jax_device_error = _probe_jax_devices(modules.get("jax"))
    nvidia_smi = run_command(NVIDIA_SMI_QUERY)
    nvidia_gpus = _parse_nvidia_smi_csv(str(nvidia_smi.get("stdout") or ""))

    failures: list[str] = []
    if len(visible_devices) != 1:
        failures.append("cuda_visible_devices_count")
    for name in REQUIRED_IMPORTS:
        if not import_status[name]["available"]:
            failures.append(f"import:{name}")
    if jax_device_error is not None:
        failures.append("jax_devices")
    elif not jax_devices:
        failures.append("jax_devices")
    elif not any(_is_gpu_device(device) for device in jax_devices):
        failures.append("jax_gpu_device")
    if nvidia_smi.get("returncode") != 0 or nvidia_smi.get("error"):
        failures.append("nvidia_smi")
    fragment = required_gpu_name_fragment.strip().lower()
    if fragment and not _detected_gpu_name_contains(fragment, jax_devices, nvidia_gpus):
        failures.append("gpu_name_fragment")

    return {
        "schema_version": 1,
        "passed": not failures,
        "failures": failures,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "required_gpu_name_fragment": required_gpu_name_fragment,
        "cuda_visible_devices": visible_devices,
        "imports": import_status,
        "jax_devices": jax_devices,
        "jax_device_error": jax_device_error,
        "nvidia_smi": {
            **nvidia_smi,
            "gpus": nvidia_gpus,
        },
    }


def run_nvidia_smi(command: tuple[str, ...]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
        }
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "error": None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(required_gpu_name_fragment=args.required_gpu_name_fragment)
    indent = 2 if args.pretty else None
    text = json.dumps(report, indent=indent, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    print(text)
    return 0 if report["passed"] else 1


def _visible_cuda_devices(environ: dict[str, str]) -> list[str]:
    value = environ.get("CUDA_VISIBLE_DEVICES", "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _probe_imports(import_module) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    modules: dict[str, Any] = {}
    for name in REQUIRED_IMPORTS:
        try:
            module = import_module(name)
        except Exception as exc:
            status[name] = {
                "available": False,
                "version": None,
                "error": str(exc),
            }
            continue
        modules[name] = module
        status[name] = {
            "available": True,
            "version": getattr(module, "__version__", None),
            "error": None,
        }
    return status, modules


def _probe_jax_devices(jax_module: Any | None) -> tuple[list[dict[str, Any]], str | None]:
    if jax_module is None:
        return [], "jax import unavailable"
    try:
        devices = jax_module.devices()
    except Exception as exc:
        return [], str(exc)
    return [_jax_device_record(device) for device in devices], None


def _jax_device_record(device: Any) -> dict[str, Any]:
    return {
        "id": getattr(device, "id", None),
        "platform": getattr(device, "platform", None),
        "device_kind": getattr(device, "device_kind", None),
        "repr": repr(device),
    }


def _parse_nvidia_smi_csv(stdout: str) -> list[dict[str, str]]:
    gpus = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            gpus.append({"raw": line.strip()})
            continue
        index, name, memory_total, memory_used, driver_version = parts[:5]
        gpus.append(
            {
                "index": index,
                "name": name,
                "memory_total": memory_total,
                "memory_used": memory_used,
                "driver_version": driver_version,
            }
        )
    return gpus


def _is_gpu_device(device: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(value).lower()
        for value in (
            device.get("platform"),
            device.get("device_kind"),
            device.get("repr"),
        )
        if value is not None
    )
    return "gpu" in haystack or "cuda" in haystack


def _detected_gpu_name_contains(
    fragment: str,
    jax_devices: list[dict[str, Any]],
    nvidia_gpus: list[dict[str, str]],
) -> bool:
    for device in jax_devices:
        haystack = " ".join(
            str(value).lower()
            for value in (device.get("device_kind"), device.get("repr"))
            if value is not None
        )
        if fragment in haystack:
            return True
    for gpu in nvidia_gpus:
        if fragment in gpu.get("name", "").lower():
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
