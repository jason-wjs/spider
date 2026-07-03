"""Motion loading and resampling for G1 WBC tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    COMMAND_BODY_NAMES,
    ISAACLAB_TO_MUJOCO_BODY_REINDEX,
    ISAACLAB_TO_MUJOCO_JOINT_REINDEX,
    LEFT_FOOT_BODY_NAME,
    MUJOCO_BODY_NAMES,
    POLICY_DT,
    QPOS_DIM,
    QVEL_DIM,
    RIGHT_FOOT_BODY_NAME,
)
from spider.tasks.g1_wbc.math_utils import (
    axis_angle_from_quat,
    finite_difference_root_velocity,
    quat_inv,
    quat_mul,
    quat_error_magnitude,
    world_velocity_to_qvel,
)

MotionType = Literal["auto", "mujoco", "isaaclab"]


@dataclass
class G1Motion:
    path: Path
    motion_type: Literal["mujoco", "isaaclab"]
    fps: float
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    contact: torch.Tensor

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def device(self) -> torch.device:
        return self.joint_pos.device

    @property
    def body_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(MUJOCO_BODY_NAMES)}

    @property
    def command_body_indices(self) -> torch.Tensor:
        return torch.tensor(
            [self.body_index[name] for name in COMMAND_BODY_NAMES],
            dtype=torch.long,
            device=self.device,
        )

    def qpos(self) -> torch.Tensor:
        root_pos = self.body_pos_w[:, 0]
        root_quat = self.body_quat_w[:, 0]
        return torch.cat([root_pos, root_quat, self.joint_pos], dim=-1)

    def qvel(self) -> torch.Tensor:
        root_world_vel = torch.cat(
            [self.body_lin_vel_w[:, 0], self.body_ang_vel_w[:, 0]], dim=-1
        )
        root_qvel = world_velocity_to_qvel(self.qpos()[:, :7], root_world_vel)
        return torch.cat([root_qvel, self.joint_vel], dim=-1)

    def command_tensor(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=-1)

    def command_bodies(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.command_body_indices
        return (
            self.body_pos_w[:, idx],
            self.body_quat_w[:, idx],
            self.body_lin_vel_w[:, idx],
            self.body_ang_vel_w[:, idx],
        )

    def to(self, device: str | torch.device) -> "G1Motion":
        return G1Motion(
            path=self.path,
            motion_type=self.motion_type,
            fps=self.fps,
            joint_pos=self.joint_pos.to(device),
            joint_vel=self.joint_vel.to(device),
            body_pos_w=self.body_pos_w.to(device),
            body_quat_w=self.body_quat_w.to(device),
            body_lin_vel_w=self.body_lin_vel_w.to(device),
            body_ang_vel_w=self.body_ang_vel_w.to(device),
            contact=self.contact.to(device),
        )


@dataclass
class G1CommandBatch:
    path: Path
    motion_type: Literal["mujoco", "isaaclab"]
    fps: float
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    qpos_trajectory: torch.Tensor
    qvel_trajectory: torch.Tensor

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.joint_pos.shape[1])

    @property
    def device(self) -> torch.device:
        return self.joint_pos.device

    @property
    def body_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(MUJOCO_BODY_NAMES)}

    def to(self, device: str | torch.device) -> "G1CommandBatch":
        return G1CommandBatch(
            path=self.path,
            motion_type=self.motion_type,
            fps=self.fps,
            joint_pos=self.joint_pos.to(device),
            joint_vel=self.joint_vel.to(device),
            body_pos_w=self.body_pos_w.to(device),
            body_quat_w=self.body_quat_w.to(device),
            body_lin_vel_w=self.body_lin_vel_w.to(device),
            body_ang_vel_w=self.body_ang_vel_w.to(device),
            qpos_trajectory=self.qpos_trajectory.to(device),
            qvel_trajectory=self.qvel_trajectory.to(device),
        )


def detect_motion_type(path: Path, raw: np.lib.npyio.NpzFile) -> Literal["mujoco", "isaaclab"]:
    if "motion_type" in raw.files:
        value = raw["motion_type"]
        if value.shape == ():
            text = str(value.item()).lower()
        else:
            text = str(value.tolist()).lower()
        if "mujoco" in text:
            return "mujoco"
        if "isaac" in text:
            return "isaaclab"
    name = path.name.lower()
    if "mujoco" in name:
        return "mujoco"
    if "isaac" in name or "isaaclab" in name:
        return "isaaclab"
    return "isaaclab"


def load_motion(
    motion_path: str | Path,
    *,
    motion_type: MotionType = "auto",
    device: str | torch.device = "cpu",
    target_dt: float = POLICY_DT,
) -> G1Motion:
    path = Path(motion_path).expanduser().resolve()
    raw = np.load(path)
    resolved_type = detect_motion_type(path, raw) if motion_type == "auto" else motion_type
    if resolved_type == "auto":
        raise ValueError("motion_type must resolve to 'mujoco' or 'isaaclab'.")

    required = [
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]
    missing = [key for key in required if key not in raw.files]
    if missing:
        raise ValueError(f"Motion file {path} is missing keys: {missing}")

    fps = _fps_from_raw(raw, target_dt=target_dt)
    joint_pos = torch.tensor(raw["joint_pos"], dtype=torch.float32, device=device)
    joint_vel = torch.tensor(raw["joint_vel"], dtype=torch.float32, device=device)
    body_pos_w = torch.tensor(raw["body_pos_w"], dtype=torch.float32, device=device)
    body_quat_w = torch.tensor(raw["body_quat_w"], dtype=torch.float32, device=device)
    body_lin_vel_w = torch.tensor(
        raw["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    body_ang_vel_w = torch.tensor(
        raw["body_ang_vel_w"], dtype=torch.float32, device=device
    )

    if resolved_type == "isaaclab":
        joint_pos = joint_pos[:, ISAACLAB_TO_MUJOCO_JOINT_REINDEX]
        joint_vel = joint_vel[:, ISAACLAB_TO_MUJOCO_JOINT_REINDEX]
        body_pos_w = body_pos_w[:, ISAACLAB_TO_MUJOCO_BODY_REINDEX]
        body_quat_w = body_quat_w[:, ISAACLAB_TO_MUJOCO_BODY_REINDEX]
        body_lin_vel_w = body_lin_vel_w[:, ISAACLAB_TO_MUJOCO_BODY_REINDEX]
        body_ang_vel_w = body_ang_vel_w[:, ISAACLAB_TO_MUJOCO_BODY_REINDEX]

    motion = G1Motion(
        path=path,
        motion_type=resolved_type,
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        contact=torch.empty((joint_pos.shape[0], 2), device=device),
    )
    motion = resample_motion(motion, target_dt=target_dt)
    contact = estimate_foot_contacts(motion)
    return G1Motion(
        path=motion.path,
        motion_type=motion.motion_type,
        fps=1.0 / target_dt,
        joint_pos=motion.joint_pos,
        joint_vel=motion.joint_vel,
        body_pos_w=motion.body_pos_w,
        body_quat_w=motion.body_quat_w,
        body_lin_vel_w=motion.body_lin_vel_w,
        body_ang_vel_w=motion.body_ang_vel_w,
        contact=contact,
    )


def _fps_from_raw(raw: np.lib.npyio.NpzFile, *, target_dt: float) -> float:
    if "fps" not in raw.files:
        return 1.0 / target_dt
    value = np.asarray(raw["fps"])
    if value.size == 0:
        return 1.0 / target_dt
    if value.size != 1:
        raise ValueError(f"Expected fps to be scalar or length 1, got shape {value.shape}.")
    return float(value.reshape(-1)[0])


def _slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    small = dot > 0.9995
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp(min=1e-8)
    theta = theta_0 * alpha
    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    out = s0 * q0 + s1 * q1
    lerp = q0 + alpha * (q1 - q0)
    out = torch.where(small, lerp, out)
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _resample_linear(x: torch.Tensor, src_dt: float, target_dt: float) -> torch.Tensor:
    if x.shape[0] <= 1 or abs(src_dt - target_dt) < 1e-7:
        return x
    duration = (x.shape[0] - 1) * src_dt
    out_len = int(np.floor(duration / target_dt + 1e-6)) + 1
    t = torch.arange(out_len, device=x.device, dtype=x.dtype) * target_dt
    u = (t / src_dt).clamp(max=x.shape[0] - 1)
    i0 = torch.floor(u).long()
    i1 = torch.clamp(i0 + 1, max=x.shape[0] - 1)
    a = (u - i0.to(u.dtype)).view(-1, *([1] * (x.ndim - 1)))
    return x[i0] * (1.0 - a) + x[i1] * a


def _resample_quat(x: torch.Tensor, src_dt: float, target_dt: float) -> torch.Tensor:
    if x.shape[0] <= 1 or abs(src_dt - target_dt) < 1e-7:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    duration = (x.shape[0] - 1) * src_dt
    out_len = int(np.floor(duration / target_dt + 1e-6)) + 1
    t = torch.arange(out_len, device=x.device, dtype=x.dtype) * target_dt
    u = (t / src_dt).clamp(max=x.shape[0] - 1)
    i0 = torch.floor(u).long()
    i1 = torch.clamp(i0 + 1, max=x.shape[0] - 1)
    a = (u - i0.to(u.dtype)).view(-1, *([1] * (x.ndim - 1)))
    return _slerp(x[i0], x[i1], a)


def resample_motion(motion: G1Motion, *, target_dt: float = POLICY_DT) -> G1Motion:
    src_dt = 1.0 / float(motion.fps)
    if abs(src_dt - target_dt) < 1e-7:
        return motion
    joint_pos = _resample_linear(motion.joint_pos, src_dt, target_dt)
    body_pos_w = _resample_linear(motion.body_pos_w, src_dt, target_dt)
    body_quat_w = _resample_quat(motion.body_quat_w, src_dt, target_dt)
    body_lin_vel_w = _resample_linear(motion.body_lin_vel_w, src_dt, target_dt)
    body_ang_vel_w = _resample_linear(motion.body_ang_vel_w, src_dt, target_dt)
    joint_vel = _resample_linear(motion.joint_vel, src_dt, target_dt)
    if torch.allclose(joint_vel, torch.zeros_like(joint_vel)) and joint_pos.shape[0] > 1:
        joint_vel = torch.zeros_like(joint_pos)
        joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) / target_dt
        joint_vel[-1] = joint_vel[-2]
    root_vel = finite_difference_root_velocity(
        torch.cat([body_pos_w[:, 0], body_quat_w[:, 0]], dim=-1), target_dt
    )
    if torch.allclose(body_lin_vel_w, torch.zeros_like(body_lin_vel_w)):
        body_lin_vel_w = body_lin_vel_w.clone()
        body_lin_vel_w[:, 0] = root_vel[:, :3]
    if torch.allclose(body_ang_vel_w, torch.zeros_like(body_ang_vel_w)):
        body_ang_vel_w = body_ang_vel_w.clone()
        body_ang_vel_w[:, 0] = root_vel[:, 3:]
    return G1Motion(
        path=motion.path,
        motion_type=motion.motion_type,
        fps=1.0 / target_dt,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        contact=torch.empty((joint_pos.shape[0], 2), device=joint_pos.device),
    )


def estimate_foot_contacts(
    motion: G1Motion,
    *,
    height_threshold: float = 0.055,
    speed_threshold: float = 0.35,
) -> torch.Tensor:
    body_index = motion.body_index
    foot_ids = [body_index[LEFT_FOOT_BODY_NAME], body_index[RIGHT_FOOT_BODY_NAME]]
    pos = motion.body_pos_w[:, foot_ids]
    vel = motion.body_lin_vel_w[:, foot_ids]
    floor_z = torch.quantile(pos[..., 2].reshape(-1), 0.02)
    height = pos[..., 2] - floor_z
    speed = torch.linalg.norm(vel, dim=-1)
    contact = (height < height_threshold) & (speed < speed_threshold)
    contact_f = contact.to(torch.float32)
    if contact_f.shape[0] >= 3:
        prev = contact_f[:-2]
        cur = contact_f[1:-1]
        nxt = contact_f[2:]
        filtered = torch.where(prev.eq(nxt), prev, cur)
        contact_f = torch.cat([contact_f[:1], filtered, contact_f[-1:]], dim=0)
    return contact_f


def validate_motion_dims(motion: G1Motion) -> None:
    if motion.joint_pos.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM} joint positions, got {motion.joint_pos.shape}.")
    if motion.body_pos_w.shape[1] != len(MUJOCO_BODY_NAMES):
        raise ValueError(
            f"Expected {len(MUJOCO_BODY_NAMES)} bodies, got {motion.body_pos_w.shape}."
        )
    if motion.qpos().shape[-1] != QPOS_DIM or motion.qvel().shape[-1] != QVEL_DIM:
        raise ValueError("Motion qpos/qvel dimensions do not match G1 29dof model.")


def root_pose_error(a_qpos: torch.Tensor, b_qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.linalg.norm(a_qpos[..., :3] - b_qpos[..., :3], dim=-1)
    rot = quat_error_magnitude(a_qpos[..., 3:7], b_qpos[..., 3:7])
    return pos, rot


def qvel_from_qpos_trajectory(qpos: torch.Tensor, dt: float = POLICY_DT) -> torch.Tensor:
    """Finite-difference a qpos trajectory into MuJoCo qvel convention."""

    if qpos.ndim == 2:
        qpos_batched = qpos[:, None, :]
        squeeze = True
    elif qpos.ndim == 3:
        qpos_batched = qpos
        squeeze = False
    else:
        raise ValueError(f"Expected qpos shape (T, 36) or (T, N, 36), got {qpos.shape}")
    if qpos_batched.shape[-1] != QPOS_DIM:
        raise ValueError(f"Expected qpos dim {QPOS_DIM}, got {qpos_batched.shape}")

    qvel = torch.zeros(
        qpos_batched.shape[:-1] + (QVEL_DIM,),
        dtype=qpos_batched.dtype,
        device=qpos_batched.device,
    )
    if qpos_batched.shape[0] <= 1:
        return qvel[:, 0] if squeeze else qvel

    lin_vel = torch.zeros_like(qvel[..., :3])
    ang_vel_w = torch.zeros_like(qvel[..., :3])
    joint_vel = torch.zeros_like(qpos_batched[..., 7:])

    lin_vel[:-1] = (qpos_batched[1:, :, :3] - qpos_batched[:-1, :, :3]) / dt
    lin_vel[-1] = lin_vel[-2]
    delta_quat = quat_mul(qpos_batched[1:, :, 3:7], quat_inv(qpos_batched[:-1, :, 3:7]))
    ang_vel_w[:-1] = axis_angle_from_quat(delta_quat) / dt
    ang_vel_w[-1] = ang_vel_w[-2]
    joint_vel[:-1] = (qpos_batched[1:, :, 7:] - qpos_batched[:-1, :, 7:]) / dt
    joint_vel[-1] = joint_vel[-2]

    root_world_vel = torch.cat([lin_vel, ang_vel_w], dim=-1)
    qvel[..., :6] = world_velocity_to_qvel(qpos_batched[..., :7], root_world_vel)
    qvel[..., 6:] = joint_vel
    return qvel[:, 0] if squeeze else qvel
